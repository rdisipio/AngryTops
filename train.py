#!/usr/bin/env python

import os, sys

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import sklearn.utils

from ROOT import *

try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
np.set_printoptions( precision=2, suppress=True )

import pandas as pd

from features import *
import models

early_stopping = EarlyStopping( monitor='val_loss', min_delta=0.01, patience=3, mode='min' )
callbacks_list = [ early_stopping ]

training_filename = "csv/training.csv"
if len(sys.argv) > 1: training_filename = sys.argv[1]

# read in input file
data = pd.read_csv( training_filename, delimiter=',', names=header )

#input_features_t_lep  = input_features_t_lep
#input_features_t_had  = input_features_t_had
#models.n_cols_t_lep   = n_features_per_jet
#models.n_rows_t_lep   = n_rows_t_lep
#models.n_cols_t_had   = n_features_per_jet
#models.n_rows_t_had   = n_rows_t_had
#target_features          = target_features_ttbar
#models.n_target_features = len(target_features)

#print "INFO: input features:"
#print input_features

#X_train_t_lep = data[input_features_t_lep].values
#X_train_t_had = data[input_features_t_had].values
#y_train = data[target_features].values
#print "INFO: target hadronic and leptonic tops four-momenta:"
#print y_train

X_jets   = data[input_features_jets].values
X_lepton = data[input_features_lep].values

y_W_lep = data[target_features_W_lep].values
y_W_had = data[target_features_W_had].values
y_t_lep = data[target_features_t_lep].values
y_t_had = data[target_features_t_had].values

event_info = data[features_event_info].values
n_events   = len(event_info)

# standardize input and target
#X_t_lep_scaler = StandardScaler()
#X_t_had_scaler = StandardScaler()
#y_scaler = StandardScaler()

#X_train_t_lep = X_t_lep_scaler.fit_transform( X_train_t_lep )
#X_train_t_had = X_t_had_scaler.fit_transform( X_train_t_had )
#y_train = y_scaler.fit_transform( y_train )

# reshape input
#X_train_t_lep = X_train_t_lep.reshape( (n_events, models.n_rows_t_lep, models.n_cols_t_lep) )
#X_train_t_had = X_train_t_had.reshape( (n_events, models.n_rows_t_had, models.n_cols_t_had) )
X_jets   = X_jets.reshape( (n_events,n_jets_per_event,n_features_per_jet) )
#X_lepton = X_lep.reshape( (n_events,1,6) )

#print "INFO: input shape (t lep):", X_train_t_lep.shape
#print "INFO: input shape (t had):", X_train_t_had.shape
#print "INFO: target shape:", y_train.shape

# use event weights?
mc_weights = None
#mc_weights = data["weight"].values

# training parameters
MAX_EPOCHS = 50
BATCH_SIZE = 2048

# go on with the training
dnn = KerasRegressor( build_fn=models.create_model_multi,
                      epochs=MAX_EPOCHS, batch_size=BATCH_SIZE
                     )

#dnn.fit( [ X_train_t_lep,X_train_t_had], y_train,
dnn.fit( { 'jets_input':X_jets, 'lepton_input':X_lepton },
         { 'W_lep_out':y_W_lep, 'W_had_out':y_W_had,
           'b_lep_out':y_b_lep, 'b_had_out':y_b_had,
           't_lep_out':y_t_lep, 't_had_out':y_t_had},
         validation_split=0.10, shuffle=True,
         callbacks=callbacks_list,
         sample_weight=mc_weights,  
         verbose=1 )


model_filename = "keras/model.rnn.PxPyPzEMBw.h5" 
dnn.model.save( model_filename )
print "INFO: model saved to file:  ", model_filename

#scaler_filename = "keras/scaler.rnn.PxPyPzEMBw.pkl"
#with open( scaler_filename, "wb" ) as file_scaler:
#  pickle.dump( X_t_lep_scaler, file_scaler )
#  pickle.dump( X_t_had_scaler, file_scaler )
#  pickle.dump( y_scaler, file_scaler )
#print "INFO: scalers saved to file:", scaler_filename

  
