from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization


"""
https://stackoverflow.com/questions/48491737/understanding-keras-lstms-role-of-batch-size-and-statefulness

inputs: A 3D tensor, with shape [batch, timesteps, feature].
units: Positive integer, dimensionality of the output space.
return_sequences: Whether to return the last output in the output sequence, or the full sequence.
"""
def create_model(activation='tanh', lr=1e-3, reg=0.0, dropout=0.0, num_units=300, n_steps_in=1, n_steps_out=600):

  model = Sequential()
  model.add(
      GRU( 
          units = num_units, 
          activation = activation,
          recurrent_activation = activation,
          dropout = dropout, 
          kernel_regularizer = l2(reg), 
          return_sequences=True, 
          )
      )

  model.add(
      GRU(
          units=num_units, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg),
          #return_sequences = True,
          )
      )
  #?, 1200, 546
  
  model.add(Dense(units=n_steps_out, kernel_regularizer = l2(reg)))
  model.compile(optimizer='adam', loss='mse', metrics='mae')
  
  return model