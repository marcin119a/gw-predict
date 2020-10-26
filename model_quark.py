from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras.optimizers as optim
from keras.regularizers import l1,l2
import numpy as np
from lstm_quark import create_model, X_train, X_test, y_test, y_train
from datetime import datetime


model = create_model(activation='tanh', lr=9.345405211822168 * (10**-5), reg=3.564391979952528 * (10** -5), dropout= 0.369863105580737, num_layers=546)
stats = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=100, batch_size=32)


y_pred = model.predict(X_test)
date = datetime.today().strftime("%Y-%m-%d")
filename = f'model/model{date}.h5'
model.save(filename, save_format='h5')

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(stats.history['loss'], label='train')
ax.plot(stats.history['val_loss'], label='test')
ax.set(xlabel='Epochs', ylabel='MSE Loss', xticks=np.arange(0, 100, 5))
ax.legend()
fig.tight_layout()

fig.savefig('loss_vs_val_quark.png')

y_test = np.squeeze(y_test)
y_pred = np.squeeze(y_pred)

print('RMSE: {}'.format(np.sqrt(np.mean((y_test - y_pred)**2))))
