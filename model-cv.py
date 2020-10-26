
import numpy as np
import hickle as hkl
import keras.optimizers as optim

from sklearn.model_selection import KFold, cross_val_score
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.regularizers import l1,l2
from matplotlib import pyplot
import tensorflow as tf

array_hkl = hkl.load('data/D-SET-norm(100,1200).hkl')
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
  model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics = [tf.keras.metrics.MeanSquaredError()])
  
  return model

# remove one dimension
y_test = np.squeeze(y_test)
y_train = np.squeeze(y_train)

   
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std
from scipy.stats import sem

def evaluate_model(X,y, repeats):
    params = {
        'activation':'tanh', 
        'lr':9.345405211822168 * (10**-5), 
        'reg':3.564391979952528 * (10** -5),
        'dropout': 0.369863105580737, 
        'number_layers':546
    }

    # prepare the cross-validation procedure
    classifier = KerasRegressor(build_fn = create_model, **params, batch_size = 10, epochs = 20)

    accuracies = cross_val_score(estimator = classifier, X = np.vstack((X_train, X_test)), y = np.vstack((y_train, y_test)), cv = 10)

    return accuracies

# configurations for test
repeats = range(1)
results = list()
for r in repeats:
	# evaluate using a given number of repeats
	accuracies = evaluate_model(X_test, y_test, r)
	# summarize
	print('>%d mean=%.4f se=%.3f' % (r, mean(accuracies), sem(accuracies)))
	# store
	results.append(accuracies)
# plot the results
pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
pyplot.show()
pyplot.savefig('boxplot.png')