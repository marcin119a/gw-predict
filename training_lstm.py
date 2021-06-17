from generator import split_dataset, random_dataset
import keras.optimizers as optim
import numpy as np
from model_lstm import create_model
import argparse
import tensorflow as tf
import mlflow
from utilities import split_dataset as split_data


#import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def model_run(file_name, activation='tanh', lr=0.001,
             reg=0.002, dropout=0, num_units=500,
             epochs=100, batch_size=200,
             m1=30, m2=60, ts_lenght=800
             ):

    #X, y = random_dataset(m1=m1, m2=m2, n_steps=ts_lenght, batch_size=batch_size, channels=3)
    #X_test, X_train, y_test, y_train = split_dataset(X, y)
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


    print('RMSE: {}'.format(np.sqrt(np.mean((y_test - y_pred)**2))))
    

    return model, val_loss, np.sqrt(np.mean((y_test - y_pred)**2)), stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", type=str, default='D-SET(n=1000,time_steps=800).hkl', help="File")
    ap.add_argument("-act", "--activation", type=str, default='tanh', help="Activation")
    ap.add_argument("-lr", "--lr", type=int, default=0.0001, help="Learning Rate")
    ap.add_argument("-reg", "--reg", type=int, default=0, help="Regularizaction")
    ap.add_argument("-dropout", "--dropout", type=int, default=0.37, help="Dropout")
    ap.add_argument("-nu", "--num_units", type=int, default=300, help="Num of units for first layer RNN")
    ap.add_argument("-ts", "--ts_lenght", type=bool, default=800, help="Time series lenght")
    ap.add_argument("-epochs", "--num_epoch", type=int, default=100, help="Epochs")
    ap.add_argument("-bs", "--batch_size", type=int, default=100, help="Batch size")
    ap.add_argument("-m1", "--mass1", type=int, default=30, help="Mass of first black hole")
    ap.add_argument("-m2", "--mass2", type=int, default=60, help="Mass of first black hole")

    
    args = vars(ap.parse_args())

    params = {
        'file_name': args['file'],
        'activation': args['activation'],
        'lr' : args['lr'], 
        'reg' : args['reg'], 
        'dropout' : args['dropout'], 
        'num_units' :  args['num_units'],
        'epochs': args['num_epoch'],
        'ts_lenght': args['ts_lenght'],
        'm1':  args['mass1'],
        'm2':  args['mass2'],
    }
    
    with mlflow.start_run():

        model, val_loss, history = model_run(**params)
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        for step, (mloss, mvloss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            metrics = {'loss': float(mloss), 'val_loss': float(mvloss)}
            mlflow.log_metrics(metrics, step=step)


        mlflow.keras.log_model(model, "lstm_pycbc")

#https://github.com/gwastro/pycbc/blob/master/pycbc/filter/matchedfilter.py