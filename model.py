from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras.optimizers as optim
from keras.regularizers import l1,l2
import hickle as hkl
import numpy as np
from lstm import create_model, X_train, X_test, y_test, y_train


model = create_model(activation = 'tanh', dropout = 0.03798050443750123, lr = 4.217325916228721 * (10** -6), reg = 0.00022299214636513958)

stats = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=20, batch_size=32)
                  
y_pred = model.predict(X_test)

model.save('model/model2020-10-29.h5', save_format='h5')

import matplotlib.pyplot as plt


fig, ax = plt.subplots()


ax.plot(stats.history['loss'], label='train')
ax.plot(stats.history['val_loss'], label='test')
ax.set(xlabel='Epochs', ylabel='MSE Loss', xticks=np.arange(0, 21, 5))
ax.legend()
fig.tight_layout()

fig.savefig('loss_vs_val_standrad.png')

y_test = np.squeeze(y_test)
y_pred = np.squeeze(y_pred)

print('RMSE: {}'.format(np.sqrt(np.mean((y_test - y_pred)**2))))
