from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
# define min max scaler

import hickle as hkl


file = ''

array_hkl = hkl.load(file)
X_train = array_hkl.get('xtrain')
X_test = array_hkl.get('xtest')
y_train = array_hkl.get('ytrain')
y_test = array_hkl.get('ytest')

scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(X)
print(scaled)
