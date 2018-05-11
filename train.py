#!/usr/bin/env python

import os, sys

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import sklearn.utils

try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
np.set_printoptions( precision=2, suppress=True )

import pandas as pd

from features import *
from models import *

early_stopping = EarlyStopping( monitor='val_loss', patience=20, mode='min' )
callbacks_list = [ early_stopping ]

training_filename = "csv/training.csv"
if len(sys.argv) > 1: training_filename = sys.argv[1]

# read in input file
data = pd.read_csv( training_filename, delimiter=',', names=header )

print "INFO: input features:"
print input_features
X_train = data[input_features].values
print "INFO: input four momenta:"
print X_train

y_train = data[target_features].values
print "INFO: target features:"
print target_features
print "INFO: target top and antitop four momenta:"
print y_train

event_info = data[['runNumber','eventNumber','weight']].values

#n = len(y_train)
#for i in range(n):
#   print event_info[i][0], event_info[i][1], y_train[i][-2], y_train[i][-1]
#print "dump ok"

# standardize input and target
X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform( X_train )
y_train_scaled = Y_scaler.fit_transform( y_train )

# use event weights?
mc_weights = None
#mc_weights = data["weight"].values

MAX_EPOCHS = 100
BATCH_SIZE = 512

dnn = KerasRegressor( build_fn=create_model_rnn, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.10, shuffle=True,
                      callbacks=callbacks_list, 
                      sample_weight=mc_weights,  
                      verbose=1 )

dnn.fit( X_train_scaled, y_train_scaled )

model_filename = "keras/model.rnn.PxPyPzEMBw.h5" 
dnn.model.save( model_filename )
print "INFO: model saved to file:  ", model_filename

scaler_filename = "keras/scaler.rnn.PxPyPzEMBw.pkl"
with open( scaler_filename, "wb" ) as file_scaler:
  pickle.dump( X_scaler, file_scaler )
  pickle.dump( y_scaler, file_scaler )
print "INFO: scalers saved to file:", scaler_filename

print "INFO: testing..."

y_fitted = dnn.predict( X_scaled )
y_fitted = y_scaler.inverse_transform( y_predicted_scaled )

# Print out example
for i in range(10):
   t_true    = TLorentzVector()
   tb_true   = TLorentzVector()
   t_fitted  = TLorentzVector()
   tb_fitted = TLorentzVector()

   t_true.SetPxPyPzE(  y_train[i][0], y_train[i][1], y_train[i][2], y_train[i][3] )
   tb_true.SetPxPyPzE( y_train[i][5], y_train[i][6], y_train[i][7], y_train[i][8] )

   t_fitted.SetPxPyPzE(  y_fitted[0][0], y_fitted[0][1], y_fitted[0][2], y_fitted[0][3] )
   tb_fitted.SetPxPyPzE( y_fitted[0][5], y_fitted[0][6], y_fitted[0][7], y_fitted[0][8] )

   print "en=%i) Top :: true=(%.1f, %.2f, %.2f, %.1f ; %.1f) :: fitted=(%.1f, %.2f, %.2f, %.1f ; %.1f)" % \
              ( t_true.Pt(),   t_true.Eta(),   t_true.Phi(),   t_true.E(),   t_true.M(), \
                t_fitted.Pt(), t_fitted.Eta(), t_fitted.Phi(), t_fitted.E(), t_fitted.M() )
   print "en=%i) Anti-Top :: true=(%.1f, %.2f, %.2f, %.1f ; %.1f) :: fitted=(%.1f, %.2f, %.2f, %.1f ; %.1f)" % \
              ( tb_true.Pt(),   tb_true.Eta(),   tb_true.Phi(),   tb_true.E(),   tb_true.M(), \
                tb_fitted.Pt(), tb_fitted.Eta(), tb_fitted.Phi(), tb_fitted.E(), tb_fitted.M() )

