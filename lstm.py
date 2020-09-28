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

def create_model(activation='tanh', lr=1e-3, reg=0.0, dropout=0.0):
  n_steps_in, n_steps_out = X_train.shape[1], 2
  model = Sequential()
  model.add(
      LSTM(
          200, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg), 
          return_sequences=True, 
          input_shape=(n_steps_in, 1))
      )
  model.add(
      LSTM(200, 
          activation=activation, 
          dropout = dropout, 
          kernel_regularizer = l2(reg))
      )
  model.add(Dense(n_steps_out, kernel_regularizer = l2(reg)))
  model.compile(optimizer='adam', loss='mse')
  
  return model


# define a search space
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

space = {
    'activation':hp.choice('activation', ('relu', 'tanh')),
    'lr':hp.loguniform('lr', np.log(1e-6), np.log(1e-2)), 
    'dropout':hp.uniform('dropout', 0.0, 1.0), 
    'reg':hp.uniform('reg', 1e-6, 1e-3)
}

# define loss function
def loss(params):
    
    model = create_model(**params)

    _ = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
    val_loss = model.evaluate(X_test, y_test, verbose=1)
    print('Loss: {}'.format(val_loss))
    
    return {'loss':val_loss, 'status':STATUS_OK}


print('Begin tuning')
print('------------')
trials = Trials()
best_params = fmin(loss,
                   space = space,
                   algo = tpe.suggest,
                   max_evals = 20,
                   trials = trials)
print('')
print('Best parameters:')
print('----------------')
best_params['activation'] = ['relu', 'tanh'][best_params['activation']]
for k, v in best_params.items():
    print('{} = {}'.format(k, v))