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

def create_model_rnn( dropout=0 ):
   inshape   = ( (1+n_jets_per_event), n_features_per_jet )

   input = Input( shape=inshape )

   x = TimeDistributed( Dense(200), input_shape=inshape )(input)
   x = LSTM( 100, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  80, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  50, recurrent_dropout=dropout, return_sequences=False )(x)

   x = Dense(30)(x)
   x = Dense(20)(x)
   
   output = Dense( 2*n_features_per_top )(x)

   model = Model(inputs=[input], outputs=[output] )

   model.compile( optimizer='adam', loss='mean_squared_error' )

   return model
