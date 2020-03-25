# preliminary analysis

# set up
# if installed, keras uses tf as backend
import keras
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.optimizers import Adagrad, Adadelta, Adam, Adamax, Nadam

from sklearn import model_selection
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score

# parameters for k-fold CV
n_folds = 10 # default is 5

# load data
X = np.genfromtxt('../data/X.csv', delimiter=',')
y_int = np.genfromtxt('../data/y_int.csv', delimiter=',')

K = len(np.unique(y_int))
P = X.shape[1]

# encode response
Y = keras.utils.to_categorical(y_int, num_classes=K)

# Function to create model
def create_base(optimizer='sgd', init='glorot_uniform',
                 n_input=64, n_hidden=64, dropout=0.5, p=1, k=1,
                 input_act='tanh', hidden_act='relu', output_act='softmax',
                 loss='categorical_crossentropy'):
    # create model w/ two hidden layers
    model = Sequential()
    model.add(Dense(n_input, activation=input_act, input_dim=p, kernel_initializer=init))
    model.add(Dropout(dropout))
    model.add(Dense(n_hidden, activation=hidden_act, kernel_initializer=init))
    model.add(Dropout(dropout))
    model.add(Dense(n_hidden, activation=hidden_act, kernel_initializer=init))
    model.add(Dropout(dropout))    
    model.add(Dense(k, activation=output_act, kernel_initializer=init))
    # Compile model
    model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    return model

# set k-fold cv
kfold = KFold(n_splits=n_folds)

# create model, categorical encoding
# TODO: larger batch size, more epochs
model = KerasClassifier(build_fn=create_base, p=P, k=K,
                        n_input=10*P, # n_hidden=128, 
                        batch_size=128, epochs=200, verbose=0)

results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# create model, integer encoding
# NB: loss='mse' but metrics='accuracy'
m_int = KerasClassifier(build_fn=create_base, p=P, k=1, n_input=10*P,
                        output_act='relu', loss='mse', 
                        batch_size=16, epochs=100, verbose=0)

r_int = cross_val_score(m_int, X, y_int, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (r_int.mean()*100, r_int.std()*100))

# what about sparse integer encoding?
# - loss = `sparse_categorical_crossentropy', k=K

# TODO: save models