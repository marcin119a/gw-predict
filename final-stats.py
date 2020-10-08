from tensorflow import keras
import hickle as hkl
import numpy as np 

array_hkl = hkl.load('data/D-SET(100,1200).hkl')
X_train = array_hkl.get('xtrain')
X_test = array_hkl.get('xtest')
y_train = array_hkl.get('ytrain')
y_test = array_hkl.get('ytest')

model = keras.models.load_model('/home/marcin119a/lstm-pycbc/model/model2020-10-29.h5')

X_test =  X_test.reshape(X_test.shape[0], X_test.shape[2], 1)

y_hat = model.predict(X_test)

for x in range(len(y_hat)):
  print(y_hat[x], y_test[x])

y_test = np.squeeze(y_test)

print('RMSE: {}'.format(np.sqrt(np.mean((y_test - y_hat)**2))))
