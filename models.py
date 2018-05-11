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

def create_model_rnn( n_jets_per_event=7, n_features_per_jet=6, n_features_per_top=5 ):
   inshape   = (n_jets_per_event,n_features_per_jet)

   input = Input( shape=inshape )

   x = TimeDistributed( Dense(100), input_shape=inshape )(input)
   x = LSTM(  80, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  50, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  25, recurrent_dropout=dropout, return_sequences=False )(x)

   x = Dense(20)(x)

   output = Dense( 2*n_features_per_top )(x)

   model = Model(inputs=[input], outputs=[output] )

   model.compile( optimizer='adam', loss='mean_squared_error' )

   return model
