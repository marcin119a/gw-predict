from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization


"""
https://stackoverflow.com/questions/48491737/understanding-keras-lstms-role-of-batch-size-and-statefulness
"""
def create_model(activation='tanh', lr=1e-3, reg=0.0, dropout=0.0, num_neurons=200, n_steps_in=1, n_steps_out=600):

  model = Sequential()
  #(batch_size, timesteps, units)
  #(?, 1024, 1) 
  #n_steps_in = 3
  model.add(
      GRU(
          units = num_neurons, 
          activation = activation,
          recurrent_activation = activation,
          dropout = dropout, 
          kernel_regularizer = l2(reg), 
          return_sequences=True, 
          #input_shape=(?, length of ts, 3)
          )
      )

  model.add(
      GRU(
          units=num_neurons, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg),
          #return_sequences = True,
          )
      )
  #?, 1200, 546
  
  model.add(Dense(units=n_steps_out, kernel_regularizer = l2(reg)))
  model.compile(optimizer='adam', loss='mse')
  
  return model