# prepare data for tf

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

##########################
# load complete data set
dat_all = pd.read_csv('../data/model-data-421.csv')

# explore
dat_all.head()
dat_all.columns.values.tolist()

##########################
# filter rows
dat = dat_all[(dat_all.AffectLvl.isin(['Damaged', 'Destroyed'])) & 
              # filter by building value or by assessed value?
              (dat_all.tot_bldg_v > 0) &
              # filter parcels with buildings > 0 and stories > 0
              (dat_all.total_buil > 0) & (dat_all.max_storie > 0)]

dat.shape

##########################
# drop unwanted columns
keep_cols = [
    # basic parcel info:
    'TCLDist', 'ChangeCat', 'AffectLvl', 
    # can probably drop these fixed effects:
    'subdDESC', 'TractID', 'BlockID', 'ZipCode',
    # tract-level info
    # TODO: 
    # - re-encode tract-level building ages
    'MHI', 'Pov',
    # block-level info
    # TODO: 
    # - re-encode block-level occupancy
    'Population', 'HousUnits', 'Occupied', 'Vacant', 
    'OwnWMort', 'OwnOutrigh', 'RentOcc', 'Average_ho',
    # parcel-level info from county assessor
    'tot_land_v', 'tot_bldg_v', 'assessed_v', 'total_tax', 'bldg_type', 
    'bldg_code', 'total_buil', 'max_storie', 'earliest_y','total_bldg',
]

data = dat[keep_cols]
data.shape

##########################
# select response

# 1. integer encoding
y_int = dat['EndRange'].replace(['', ' ', '-'], 4)
y_int.replace('09_2011', 0, inplace=True)
y_int.replace('10_2012', 1, inplace=True)
y_int.replace('05_2014', 2, inplace=True)
y_int.replace('02_2016', 3, inplace=True)
y_int = y_int.values.reshape(-1,1)

# 2. categorical encoding
one_hot = OneHotEncoder()
y = one_hot.fit_transform(y_int).toarray()
y.shape


######################################################################
# Alternative: re-encode existing categorical columns 
# NB: this produces some warnings about copies from slices (???)
y_cols = [
    '09_2011',
    '10_2012',
    '05_2014',
    '02_2016']

y_cat = dat[y_cols]

y_cat['sum'] = y_cat.sum(axis=1)
y_cat['None'] = np.where(y_cat['sum'] == 0, 1, 0)
y_cat.drop('sum', axis=1, inplace=True)
y_cat.shape
######################################################################


##########################
# once categorical columns have been processed
X = np.array(data)