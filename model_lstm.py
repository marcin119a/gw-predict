from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization

"""
inputs: A 3D tensor, with shape [batch, timesteps, feature].
units: Positive integer, dimensionality of the output space.
return_sequences: Whether to return the last output in the output sequence, or the full sequence.
"""
def create_model(activation='tanh', lr=1e-3, reg=0.0, dropout=0.5, num_units=200, n_steps_in=1, n_steps_out=600):
  model = Sequential()
  model.add(
      LSTM(
          units=num_units, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg), 
          return_sequences=True, 
          input_shape=(n_steps_in, 3))
      )

  model.add(
      LSTM(
          units=num_units, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg))
      )
  
  model.add(Dense(units=n_steps_out, kernel_regularizer = l2(reg)))
  model.compile(optimizer='adam', loss='mse', metrics='mae')
  
  return model