import hickle as hkl
import tensorflow as tf

from tensorflow import keras
model = keras.models.load_model('/content/lstm-pycbc/model/model2020-10-29-mass.h5')

array_hkl = hkl.load('data/D-SET(100,1200)quark.hkl')
X_train = array_hkl.get('xtrain')
X_test = array_hkl.get('xtest')
y_train = array_hkl.get('ytrain')
y_test = array_hkl.get('ytest')

X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], 1)
X_test =  X_test.reshape(X_test.shape[0], X_test.shape[2], 1)

y_train = tf.keras.utils.normalize(y_train)  
y_test = tf.keras.utils.normalize(y_test)  

y_pred = model.predict(X_test)


y_test = np.squeeze(y_test)
y_pred = np.squeeze(y_pred)


import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

data = np.random.normal(size=10000)

# plt.hist gives you the entries, edges 
# and drawables we do not need the drawables:
entries, edges, _ = plt.hist(y_test - y_pred, bins=25, range=[0.0, 0.2])

# calculate bin centers
bin_centers = 0.5 * (edges[:-1] + edges[1:])

# draw errobars, use the sqrt error. You can use what you want there
# poissonian 1 sigma intervals would make more sense
#plt.errorbar(bin_centers, entries, yerr=np.sqrt(entries), fmt='b.')
plt.savefig('histogram.png')
plt.show()
