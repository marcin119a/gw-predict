
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

random_state=42
num_folds=2

kf = KFold(n_splits=num_folds, random_state=random_state)
rstate= np.random.RandomState(random_state)


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras.optimizers as optim
from keras.regularizers import l1,l2
import hickle as hkl


array_hkl = hkl.load('data/D-SET(100,1200).hkl')
X_train = array_hkl.get('xtrain')
X_test = array_hkl.get('xtest')
y_train = array_hkl.get('ytrain')
y_test = array_hkl.get('ytest')

X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], 1)
X_test =  X_test.reshape(X_test.shape[0], X_test.shape[2], 1)

def create_model(activation='tanh', lr=1e-3, reg=0.0, dropout=0.0, number_layers=200):
  n_steps_in, n_steps_out = X_train.shape[1], 2
  model = Sequential()
  model.add(
      LSTM(
          number_layers, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg), 
          return_sequences=True, 
          input_shape=(n_steps_in, 1))
      )
  model.add(
      LSTM(
          number_layers, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg))
      )
  model.add(Dense(n_steps_out, kernel_regularizer = l2(reg)))
  model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
  
  return model

y_test = np.squeeze(y_test)
y_train = np.squeeze(y_train)

   
from keras.wrappers.scikit_learn import KerasClassifier

classifier = KerasClassifier(build_fn = create_model, batch_size = 10, epochs = 20)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
std = accuracies.std()
print(accuracies)
