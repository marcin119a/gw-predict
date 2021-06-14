from generator import random_dataset, split_dataset
import keras.optimizers as optim
import numpy as np
from model_cnn import create_model
import argparse
import tensorflow as tf
import mlflow

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def model_run(
             file_name, activation='tanh', lr= 0.001, dropout=0,
             epochs=1, batch_size=200, bn=False,
             m1=30, m2=60, ts_lenght=600
             ):
    
    X, y = random_dataset(m1=m1, m2=m2, n_steps=ts_lenght, batch_size=batch_size, channels=3)

    #(batch_size, ts_lenght, units(chanels))
    X_test, X_train, y_test, y_train = split_dataset(X, y)


    model = create_model(activation=activation, lr=lr, 
                        dropout=dropout, n_steps_out=ts_lenght, bn=bn,
                        n_steps_in=X_train.shape[1])
    
    # many-to-many rnn model
    
    y_train = tf.tile(y_train, [1, X_train.shape[1]])
    y_test = tf.tile(y_test, [1, X_test.shape[1]])


    stats = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    y_pred = model.predict(X_test)

    val_loss = model.evaluate(X_test, y_test, verbose=1)


    return model, val_loss, stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", type=str, default='data/D-SET(100,1200)-chirp.hkl', help="File")
    ap.add_argument("-act", "--activation", type=str, default='relu', help="Activation")
    ap.add_argument("-m1", "--mass1", type=int, default=30, help="Mass of first black Hole")
    ap.add_argument("-m2", "--mass2", type=int, default=50, help="Mass of second black Hole")
    ap.add_argument("-lr", "--lr", type=int, default=0.0001, help="Learning Rate")
    ap.add_argument("-dropout", "--dropout", type=int, default=0.37, help="Dropout")
    ap.add_argument("-ts", "--ts_lenght", type=int, default=600, help="Time series lenght")
    ap.add_argument("-epochs", "--num_epoch", type=int, default=100, help="Epochs")
    ap.add_argument("-bs", "--batch_size", type=int, default=100, help="Batch size")
    ap.add_argument('-bn', "--batch_normalization", type=int, default=True, help="Batch normalizaction")

    args = vars(ap.parse_args())

    params = {
        'file_name': args['file'],
        'activation': args['activation'],
        'lr' : args['lr'], 
        'dropout' : args['dropout'], 
        'epochs': args['num_epoch'],
        'batch_size':  args['batch_size'],
        'bn': args['batch_normalization'],
        'ts_lenght' :  args['ts_lenght'],
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

    
        mlflow.keras.log_model(model, "cnn_pycbc")

