from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras.optimizers as optim
from keras.regularizers import l1,l2
import hickle as hkl
import numpy as np


array_hkl = hkl.load('data/D-SET(100,1200).hkl')
X_train = array_hkl.get('xtrain')
X_test = array_hkl.get('xtest')
y_train = array_hkl.get('ytrain')
y_test = array_hkl.get('ytest')

X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], 1)
X_test =  X_test.reshape(X_test.shape[0], X_test.shape[2], 1)

def create_model(activation='tanh', lr=1e-3, reg=0.0, dropout=0.0):
  n_steps_in, n_steps_out = X_train.shape[1], 2
  model = Sequential()
  model.add(
      LSTM(
          200, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg), 
          return_sequences=True, 
          input_shape=(n_steps_in, 1))
      )
  model.add(
      LSTM(200, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg))
      )
  model.add(Dense(n_steps_out, kernel_regularizer = l2(reg)))
  model.compile(optimizer='adam', loss='mse')
  
  return model


model = create_model(activation='tanh', lr=0.41854142438805664, reg=(2.045564419075633 * (10 ** -5)), dropout=0.00015022858271954482)
stats = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=20, batch_size=32)


y_pred = model.predict(X_test)

model.save('model2020-10-29.h5', save_format='h5')

import matplotlib.pyplot as plt


fig, ax = plt.subplots()


ax.plot(stats.history['loss'], label='train')
ax.plot(stats.history['val_loss'], label='test')
ax.set(xlabel='Epochs', ylabel='MSE Loss', xticks=np.arange(0, 21, 5))
ax.legend()
fig.tight_layout()

print('RMSE: {}'.format(np.sqrt(np.mean((y_test - y_pred)**2))))
