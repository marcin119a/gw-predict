# 1D CNN neural network
# documentation: https://keras.io/layers/convolutional/
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
import tensorflow.keras.callbacks as callbacks 
from tensorflow.keras import optimizers



def create_model(time_steps=2048, num_detectors=1, num_classes=3, filters1=64, filters2=32, 
                filters3=16,  filters4=8, activation='relu',
                dropout1=0.5, dropout2=0.25, lr=0.01):
    model_m = Sequential()
    #model_m.add(Reshape((time_steps, num_detectors), input_shape=(time_steps, )))

    model_m.add(Conv1D(filters=filters1, kernel_size=16, strides=4, activation=activation))
    model_m.add(Dropout(dropout1)) 
    model_m.add(MaxPooling1D(4))

    model_m.add(Conv1D(filters=filters2, kernel_size=8, strides=2, activation=activation))
    model_m.add(Dropout(dropout1))
    model_m.add(MaxPooling1D(2))

    model_m.add(Conv1D(filters=filters3, kernel_size=8, strides=2, activation=activation))
    model_m.add(Dropout(dropout2))
    model_m.add(MaxPooling1D(2))

    model_m.add(Conv1D(filters=filters4, kernel_size=4, activation=activation))
    model_m.add(Dropout(dropout2))
    model_m.add(MaxPooling1D(2))

    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.1))

    model_m.add(Dense(num_classes, activation='softmax'))
    adam_opt = optimizers.Adam(lr=lr)

    model_m.compile(loss='categorical_crossentropy',
                    optimizer=adam_opt,
                    metrics=['accuracy'])
   
    return model_m

