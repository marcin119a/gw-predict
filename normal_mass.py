import hickle as hkl
import tensorflow as tf
import numpy as np

array_hkl = hkl.load('data/D-SET(100,1200)quark.hkl')
X_train = array_hkl.get('xtrain')
X_test = array_hkl.get('xtest')
y_train = array_hkl.get('ytrain')
y_test = array_hkl.get('ytest')



y_train = tf.keras.utils.normalize(y_train)  
y_test = tf.keras.utils.normalize(y_test)  

y_test = np.squeeze(y_test)
y_train = np.squeeze(y_train)

