from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, GlobalAveragePooling1D
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization

#(batch_size, timesteps, units)
#(?, 1024, 1) 
def create_model(activation='relu', lr=1e-3, dropout=0.5, n_steps_out=1200, bn=False, n_steps_in=1):
  model_m = Sequential()
  model_m.add(Conv1D(filters=64, kernel_size=16, strides=4, activation=activation))  
  if bn:
    model_m.add(BatchNormalization())
  model_m.add(Dropout(dropout)) 
  model_m.add(MaxPooling1D(4))

  model_m.add(Conv1D(filters=32, kernel_size=8, strides=2, activation=activation))
  if bn:
    model_m.add(BatchNormalization())
  model_m.add(Dropout(dropout))
  model_m.add(MaxPooling1D(2))

  model_m.add(GlobalAveragePooling1D())
  model_m.add(Dropout(0.1))

  model_m.add(Dense(units=n_steps_out))
  model_m.compile(optimizer='adam', loss='mse')

  return model_m