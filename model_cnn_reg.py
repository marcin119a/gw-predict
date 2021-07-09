"""
 1D CNN neural network
 documentation: https://keras.io/layers/convolutional/
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
import tensorflow.keras.callbacks as callbacks 
from tensorflow import keras
from tensorflow.keras import optimizers
import tensorflow as tf


def create_model(time_steps=2048, num_detectors=1, filters1=64, filters2=32, 
                filters3=16,  filters4=8, activation='relu',
                dropout1=0.5, dropout2=0.25, lr=0.01):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1"))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))
    
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', name="Conv1D_3"))
    
    model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', name="Dense_1"))
    model.add(Dense(1, name="Dense_2"))


    optimizer = tf.keras.optimizers.RMSprop(lr)

    model.compile(loss='mse',optimizer=optimizer, metrics=['mae'])
   
    return model

