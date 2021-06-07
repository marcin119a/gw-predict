from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras.optimizers as optim
from keras.regularizers import l1,l2
import numpy as np
from model_cnn import create_model
import argparse
import tensorflow as tf
import hickle as hkl
import mlflow
from utilities import split_dataset


def model_run(file_name, activation='tanh', lr=9.35 * (10**-5),
             reg=3.57 * (10** -5), dropout=0, num_neurons=500,
             epochs=1, batch_size=200, bn=False):
    
    X_test, X_train, y_test, y_train = split_dataset(file_name)

    y_train = tf.keras.utils.normalize(y_train)  
    y_test = tf.keras.utils.normalize(y_test)  
    y_train = tf.transpose(y_train) 
    y_test = tf.transpose(y_test)
    #(batch_size, timesteps, units)
    #(?, 1024, 1) 

    model = create_model(activation=activation, lr=lr, reg=reg, dropout=dropout, num_neurons=num_neurons, batch_normalization=bn, n_steps_in=X_train.shape[1])
    #zbudowanie modelu wiele-do-wielu 
    y_train = tf.tile(y_train, [1, X_train.shape[1]])
    y_test = tf.tile(y_test, [1, X_test.shape[1]])
    print(y_train.shape, 'y_train.shape')
    print(y_test.shape, 'y_test.shape')
    print(X_train.shape, 'X_train.shape')


    stats = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    y_pred = model.predict(X_test)
    val_loss = model.evaluate(X_test, y_test, verbose=1)

    print('RMSE: {}'.format(np.sqrt(np.mean((y_test - y_pred)**2))))


    return model, val_loss, np.sqrt(np.mean((y_test - y_pred)**2)), stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", type=str, default='data/D-SET(100,1200)-chirp.hkl', help="File")
    ap.add_argument("-act", "--activation", type=str, default='tanh', help="Activation")
    ap.add_argument("-lr", "--lr", type=bool, default=0, help="Learning Rate")
    ap.add_argument("-reg", "--reg", type=bool, default=0, help="Regularizaction")
    ap.add_argument("-dropout", "--dropout", type=bool, default=0.37, help="Dropout")
    ap.add_argument("-nn", "--num_neurons", type=bool, default=546, help="Num of neurons")
    ap.add_argument("-epochs", "--num_epoch", type=bool, default=100, help="Epochs")
    ap.add_argument("-bs", "--batch_size", type=bool, default=100, help="Batch size")
    ap.add_argument('-bn', "--batch_normalization", type=bool, default=True, help="Batch normalizaction")
    
    args = vars(ap.parse_args())

    params = {
        'file_name': args['file'],
        'activation': args['activation'],
        'lr' : args['lr'], 
        'reg' : args['reg'], 
        'dropout' : args['dropout'], 
        'num_neurons' :  args['num_neurons'],
        'epochs': args['num_epoch'],
        'batch_size':  args['batch_size'],
        'bn': args['batch_normalization']
    }
    
    #with mlflow.start_run():

    model, val_loss, rmse, history = model_run(**params)
    for key, value in params.items():
        mlflow.log_param(key, value)
    
    for step, (mloss, mvloss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        metrics = {'loss': float(mloss), 'val_loss': float(mvloss)}
        mlflow.log_metrics(metrics, step=step)

    
    mlflow.log_metric("rmse", rmse)

        #mlflow.keras.log_model(model, "cnn_pycbc")

