from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization


"""
https://stackoverflow.com/questions/48491737/understanding-keras-lstms-role-of-batch-size-and-statefulness
"""
def create_model(activation='tanh', lr=1e-3, reg=0.0, dropout=0.0, num_neurons=200, batch_normalization=False, n_steps_in=1):
  n_steps_out = 1200

  model = Sequential()
  #(batch_size, timesteps, units)
  #(?, 1024, 1) 
  #n_steps_in = 3
  model.add(
      GRU(
          units=num_neurons, 
          activation= activation,
          recurrent_activation='sigmoid',
          dropout = dropout, 
          kernel_regularizer = l2(reg), 
          return_sequences=True, 
          #input_shape=(1200, 1)
          )
      )

  if batch_normalization:
    model.add(BatchNormalization())

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
  model.add(BatchNormalization())
  
  model.add(Dense(units=n_steps_out, kernel_regularizer = l2(reg)))
  model.compile(optimizer='adam', loss='mse')
  
  return model