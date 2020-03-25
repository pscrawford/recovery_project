# prepare data for tf

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

# instantiate encoders
txt_to_int = LabelEncoder()
one_hot = OneHotEncoder()
binarizer = LabelBinarizer()

##########################
# load complete data set
dat_all = pd.read_csv('../data/model-data-421.csv')

# explore
dat_all.head()
dat_all.columns.values.tolist()

# re-econde tract-level building age
bldg_age_cols = [
    'F_05',
    'F00_04',
    'F90_00',
    'F80_90',
    'F70_80',
    'F60_70',
    'F50_60',
    'F40_50',
    'F_40'
]

# create new column of total building count in tract
dat_all['total_tract_bldgs'] = dat_all[bldg_age_cols].sum(axis=1)

# aggregate building age counts
# TODO: are these reasonable in terms of major code changes? 
dat_all['age_post_00'] = dat_all['F_05'] + dat_all['F00_04']
dat_all['age_pre_50'] = dat_all['F_40'] + dat_all['F40_50']
dat_all['age_50_80'] = dat_all['F50_60'] + dat_all['F60_70'] + dat_all['F70_80']
dat_all['age_80_00'] = dat_all['F80_90'] + dat_all['F90_00']

# re-encode block-level occupancy
# TODO...

##########################
# drop unwanted columns
keep_cols = [
    # basic parcel info:
    'TCLDist', 'ChangeCat', 'AffectLvl', 
    # NB: can probably drop these fixed effects:
    'subdDESC', 'TractID', 'BlockID', 'ZipCode',
    # tract-level info
    'age_pre_50', 'age_50_80', 'age_80_00', 'age_post_00',
    'total_tract_bldgs', 'MHI', 'Pov',
    # block-level info
    # TODO: 
    # - re-encode block-level occupancy
    'Population', 'HousUnits', 'Occupied', 'Vacant', 
    'OwnWMort', 'OwnOutrigh', 'RentOcc', 'Average_ho',
    # parcel-level info from county assessor
    'tot_land_v', 'tot_bldg_v', 'assessed_v', 'total_tax', 'bldg_type', 
    # NB: total_buil = # of buildings; total_bldg = building area
    'bldg_code', 'total_buil', 'max_storie', 'earliest_y', 'total_bldg',
]

y_cols = [
    '09_2011',
    '10_2012',
    '05_2014',
    '02_2016']

keep_cols = ['EndRange'] + y_cols + keep_cols

dat_all = dat_all[keep_cols]
dat_all.shape

##########################
# filter rows
dat = dat_all[(dat_all.AffectLvl.isin(['Damaged', 'Destroyed'])) & 
              # filter by building value or by assessed value?
              (dat_all.tot_bldg_v > 0) &
              # filter parcels with buildings > 0 and stories > 0
              (dat_all.total_buil > 0) & (dat_all.max_storie > 0)]

dat.shape

##########################
# Re-encode categorical columns:
# For ['ChangeCat', 'AffecLvl', 'subdDESC'] - pd.get_dummies
# - Drop ['TractID', 'BlockID'] ?
# For ['bldg_type', 'bldg_code'] - LabelBinarizer

cat_cols = ['ChangeCat', 'AffectLvl', 'subdDESC', 'TractID', 'BlockID']
int_cols = ['bldg_type', 'bldg_code']

# NB: returns all of dat
dat_cat = pd.get_dummies(dat, columns=cat_cols, prefix=cat_cols)
# alternative, select cat_cols?
# dat_cat = dat.select_dtypes(include=['object']).copy()

dat_int = dat[['bldg_type', 'bldg_code']].copy()

## TODO: pd.apply this method?
def int_encode(x, name):
    out = binarizer.fit_transform(x[name])
    header = [name + '_' + str(s) for s in binarizer.classes_]
    xx = pd.DataFrame(out, columns=header)
    return xx

df_list = []

for col in int_cols: 
    int_out = int_encode(dat_int, col) 
    df_list.append(int_out)
    # dat_int = pd.concat([dat_int, int_out], axis=1)
    # df_new = df_new.append(int_out, ignore_index=True)

dat_int = pd.concat(df_list, axis=1)

# NB: problems with indices because dat_cat preserves original while dat_int does not
dat_cat = pd.concat([dat_cat.reset_index(drop=True), dat_int], axis=1).drop(int_cols, axis=1)


##########################
# select response

# 1. integer encoding
# NB: clunky because of need to preserve order
# alt: Label encoding y_cols, then replace missing with len(y_cols)

# TODO: wrap in method?
# add new category (no observed recovery)
y_int = dat['EndRange'].replace(['', ' ', '-'], 4)
# encode the remaining categories as int
y_int.replace('09_2011', 0, inplace=True)
y_int.replace('10_2012', 1, inplace=True)
y_int.replace('05_2014', 2, inplace=True)
y_int.replace('02_2016', 3, inplace=True)
y_int = y_int.values.reshape(-1,1)
y_int.shape

# 2. categorical encoding
y = one_hot.fit_transform(y_int).toarray()
y.shape


######################################################################
# Alternative categorical encoding: 
# - re-encode existing categorical columns 
# NB: clunky because of sum before creating new category (no observed recovery)

y_cat = dat.loc[:,y_cols]

y_cat['sum'] = y_cat.sum(axis=1)
y_cat['None'] = np.where(y_cat['sum'] == 0, 1, 0)
y_cat.drop('sum', axis=1, inplace=True)
y_cat.shape
######################################################################


##########################
# once categorical columns have been processed
X = dat_cat.drop(['EndRange'] + y_cols, axis=1)
X = np.array(X)