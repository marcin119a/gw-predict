from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

params1 = [3,3]
params2 = [4,1]

X = np.array([x1,x2])

x1 = [np.array(x) for x in in_seq1 ]
x2 = [np.array(x) for x in in_seq2 ]

y1 = [np.array(y)  for y in params1 ]
y2 = [np.array(y) for y in params2]

X = X.reshape(X.shape[0],X.shape[1],1)
y = y.reshape(y.shape[0],y.shape[1],1)


model = Sequential()
n_features = 1
n_steps = 9

model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

model.fit(X,y)
