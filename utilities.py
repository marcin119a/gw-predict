import hickle as hkl


def split_dataset(file_name):
  array_hkl = hkl.load(file_name)
  X_train = array_hkl.get('xtrain')
  X_test = array_hkl.get('xtest')
  y_train = array_hkl.get('ytrain')
  y_test = array_hkl.get('ytest')

  X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], 1)
  X_test =  X_test.reshape(X_test.shape[0], X_test.shape[2], 1)

  return X_test, X_train, y_test, y_train

  