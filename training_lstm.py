import keras.optimizers as optim
import numpy as np
from model_lstm import create_model
import argparse
import tensorflow as tf
import mlflow
from utilities import split_dataset as split_data

#CPU training
#import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def model_run(file_name, lr,
             reg, dropout, activation, num_units,
             epochs, batch_size, m1, m2, ts_lenght
             ):

    X_test, X_train, y_test, y_train = split_data(file_name)


    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
    )

    model = create_model(activation=activation, lr=lr_schedule, reg=reg, dropout=dropout, num_units=num_units, n_steps_in=ts_lenght, n_steps_out=y_train.shape[1])
    
    y_train = tf.tile(y_train, [1, X_train.shape[1]])
    y_test = tf.tile(y_test, [1, X_test.shape[1]])

    stats = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    
    y_pred = model.predict(X_test)
    val_loss = model.evaluate(X_test, y_test, verbose=1)


    return model, val_loss, stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", type=str, default='D-SET(n=1000,ts_lenght=800,m1=50,m1=80).hkl', help="File")
    ap.add_argument("-act", "--activation", type=str, default='tanh', help="Activation")
    ap.add_argument("-lr", "--lr", type=float, default=0.0001, help="Learning Rate")
    ap.add_argument("-reg", "--reg", type=float, default=0, help="Regularizaction")
    ap.add_argument("-dropout", "--dropout", type=float, default=0.5, help="Dropout")
    ap.add_argument("-nu", "--num_units", type=int, default=300, help="Num of units for first layer RNN")
    ap.add_argument("-epochs", "--num_epoch", type=int, default=100, help="Epochs")
    ap.add_argument("-bs", "--batch_size", type=int, default=200, help="Batch size")
    ap.add_argument("-ts", "--ts_lenght", type=int, default=800, help="Time series lenght")
    ap.add_argument("-m1", "--mass1", type=int, default=50, help="Mass of first black hole")
    ap.add_argument("-m2", "--mass2", type=int, default=80, help="Mass of first black hole")

    
    args = vars(ap.parse_args())

    params = {
        'file_name': args['file'],
        'activation': args['activation'],
        'lr' : args['lr'], 
        'reg' : args['reg'], 
        'dropout' : args['dropout'], 
        'num_units' :  args['num_units'],
        'epochs': args['num_epoch'],
        'batch_size':  args['batch_size'],
        'ts_lenght': args['ts_lenght'],
        'm1':  args['mass1'],
        'm2':  args['mass2'],
    }
    
    with mlflow.start_run():

        model, val_loss, history = model_run(**params)
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        for step, (mloss, mvloss, mmae, mvmea) in enumerate(zip(history.history['loss'], 
                                                                history.history['val_loss'], 
                                                                history.history['mae'], 
                                                                history.history['val_mae'])):
            metrics = {'loss': float(mloss), 'val_loss': float(mvloss), 'mae': float(mmae), 'mvmea': float(mvmea)}
            mlflow.log_metrics(metrics, step=step)


        mlflow.keras.log_model(model, "lstm_pycbc")

#https://github.com/gwastro/pycbc/blob/master/pycbc/filter/matchedfilter.py