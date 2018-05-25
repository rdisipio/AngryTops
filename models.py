import keras
from keras.models import Model, Sequential
from keras.layers import Merge, Dense, Activation, Input, LSTM, Permute, Reshape, Masking, TimeDistributed, MaxPooling1D, Flatten
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import concatenate, maximum, dot, average
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.merge import *
from keras.optimizers import *
from keras.regularizers import l2

from features import *

n_rows_t_lep = -1
n_cols_t_lep = -1
n_rows_t_had = -1
n_cols_t_had = -1
n_target_features = -1

def create_model():
   inshape_t_lep = ( n_rows_t_lep, n_cols_t_lep )
   inshape_t_had = ( n_rows_t_had, n_cols_t_had )

   input_t_lep = Input( shape=inshape_t_lep )
   input_t_had = Input( shape=inshape_t_had )

   x_t_lep = TimeDistributed( Dense(100), input_shape=inshape_t_lep )(input_t_lep)
   x_t_lep = LSTM(  80, return_sequences=True )(x_t_lep)
   x_t_lep = LSTM(  80, return_sequences=True )(x_t_lep)
   x_t_lep = LSTM(  50, return_sequences=True )(x_t_lep)
   x_t_lep = LSTM(  50, return_sequences=False )(x_t_lep)
   x_t_lep = Dense(30)(x_t_lep)
   x_t_lep = Dense(20)(x_t_lep)
   #output_t_lep = Dense( n_target_features )(x_t_lep)

   x_t_had = TimeDistributed( Dense(100), input_shape=inshape_t_had )(input_t_had)
   x_t_had = LSTM(  80, return_sequences=True )(x_t_had)
   x_t_had = LSTM(  80, return_sequences=True )(x_t_had)
   x_t_had = LSTM(  50, return_sequences=True )(x_t_had)
   x_t_had = LSTM(  50, return_sequences=False )(x_t_had)
   x_t_had = Dense(30)(x_t_had)
   x_t_had = Dense(20)(x_t_had)
   #output_t_had = Dense( n_target_features )(x_t_had)

   x_ttbar = concatenate( [ x_t_lep, x_t_had ] )
   x_ttbar = Dense( 30 )(x_ttbar)
   x_ttbar = Dense( 20 )(x_ttbar)
   output  = Dense( n_target_features )(x_ttbar)

   model = Model(inputs=[input_t_lep, input_t_had], outputs=[output] )

   model.compile( optimizer='adam', loss='mean_squared_error' )

   return model

   
################

   
def create_model_rnn():
   inshape   = ( n_rows, n_cols )

   print "INFO: building model: input shape:", inshape
   
   input = Input( shape=inshape )
   dropout = 0.00

   x = TimeDistributed( Dense(100), input_shape=inshape )(input)

   x = LSTM(  80, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  80, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  80, recurrent_dropout=dropout, return_sequences=True )(x)

   x = LSTM(  50, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  50, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  50, recurrent_dropout=dropout, return_sequences=False )(x)

   x = Dense(30)(x)
   x = Dense(20)(x)

   output = Dense( n_target_features )(x)

   model = Model(inputs=[input], outputs=[output] )

   model.compile( optimizer='adam', loss='mean_squared_error' )

   return model
