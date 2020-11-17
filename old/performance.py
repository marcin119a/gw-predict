from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras.optimizers as optim
from keras.regularizers import l1,l2
import numpy as np
from datetime import datetime
import hickle as hkl
from tensorflow import keras


array_hkl = hkl.load('data/D-SET(100,1200).hkl')
X_train = array_hkl.get('xtrain')
X_test = array_hkl.get('xtest')
y_train = array_hkl.get('ytrain')
y_test = array_hkl.get('ytest')


X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], 1)
X_test =  X_test.reshape(X_test.shape[0], X_test.shape[2], 1)

date = '2020-10-20'
filename = f'model/model{date}.h5'

model = keras.models.load_model(filename)

y_pred = model.predict(X_test)


y_test = np.squeeze(y_test)
y_pred = np.squeeze(y_pred)

print('RMSE: {}'.format(np.sqrt(np.mean((y_test - y_pred)**2))))
