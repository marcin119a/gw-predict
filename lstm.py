from keras.initializers import glorot_normal
from keras.regularizers import l1,l2
from keras.models import Sequential
from keras.layers import (
    Dense, LSTM, RepeatVector, TimeDistributed
)

from utilities import (
  preprocessing, create_dataset, split_dataset
)

from pycbc.frame import read_frame

# Read the data directly from the Gravitational-Wave Frame (GWF) file.
file_name = "H-H1_LOSC_4_V2-1128678884-32.gwf"

# LOSC bulk data typically uses the same convention for internal channels names
# Strain is typically IFO:LOSC-STRAIN, where IFO can be H1/L1/V1.
channel_name = "H1:LOSC-STRAIN"

start = 1128678884
end = start + 32

ts = read_frame(file_name, channel_name, start, end)


dataset = preprocessing(ts)

look_back = 1
train, test = split_dataset(dataset)
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [sample, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


def create_model(in_n, activation='tanh', lr=1e-3, reg=0.0, dropout=0.0):
    """
    Return a neural network model given the hyperparameters and input shape.
    """
    n_steps_in, n_steps_out = in_n, 2
    n_features = 1

    model = Sequential()
    model.add(LSTM(32,
                   input_shape          = (n_steps_in, n_features),
                   activation           = activation,
                   kernel_regularizer   = l2(reg),
                   kernel_initializer   = glorot_normal(),
                   bias_initializer     = 'ones',
                   dropout              = dropout,
                   name                 = 'ONE'))
    model.add(LSTM(32, 
                   activation           = activation))
    model.add(Dense(n_steps_out,
                    activation          = activation,
                    kernel_regularizer  = l2(reg),
                    kernel_initializer  = glorot_normal(),
                    name                = 'TWO'))
    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model
