from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization


def create_model(activation='tanh', lr=1e-3, reg=0.0, dropout=0.0, num_layers=200, batch_normalizaction= False, n_steps_in=1):
  n_steps_out = 1
  model = Sequential()
  model.add(
      LSTM(
          num_layers, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg), 
          return_sequences=True, 
          input_shape=(n_steps_in, 1))
      )

  if batch_normalizaction:
    model.add(BatchNormalization())

  model.add(
      LSTM(
          num_layers, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg))
      )

  if batch_normalizaction:
    model.add(BatchNormalization())
  model.add(Dense(n_steps_out, kernel_regularizer = l2(reg)))
  model.compile(optimizer='adam', loss='mse')
  
  return model