from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization
import hickle as hkl

def split_dataset(file_name):
  array_hkl = hkl.load(file_name)
  X_train = array_hkl.get('xtrain')
  X_test = array_hkl.get('xtest')
  y_train = array_hkl.get('ytrain')
  y_test = array_hkl.get('ytest')

  X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], 1)
  X_test =  X_test.reshape(X_test.shape[0], X_test.shape[2], 1)

  return X_test, X_train, y_test, y_train

  

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