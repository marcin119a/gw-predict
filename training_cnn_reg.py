from utilities import split_dataset
from model_cnn_reg import create_model
import keras.optimizers as optim
import numpy as np
import argparse
import tensorflow as tf
import mlflow
import matplotlib.pyplot as plt


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def model_run(activation, lr, dropout1, dropout2, 
             filters1, filters2, filters3, filters4,
             epochs, batch_size, ts_lenght,
             ):
    file_name = 'D-SET(n=1000,ts_lenght=2048,m1=50,m1=80).hkl'

    X_test, X_train, y_test, y_train = split_dataset(file_name)

    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
    )


    model = create_model(
                    activation=activation, lr=lr, dropout1=dropout1, 
                    filters1=filters1, filters2=filters2, filters3=filters3,
                    filters4=filters4, time_steps=ts_lenght)
    
    
    # many-to-many rnn model
    args = vars(ap.parse_args())

    stats = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    y_pred = model.predict(X_test)
    print(y_pred, 'y_pred')
    print(y_test, 'y_test')
    plt.scatter(y_pred, y_test, c="g", alpha=0.5)
    plt.savefig('myfig.png')

    val_loss = model.evaluate(X_test, y_test, verbose=1)

    return model, val_loss, stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-act", "--activation", type=str, default='tanh', help="Activation")
    ap.add_argument("-lr", "--lr", type=float, default=0.001, help="Learning Rate")
    ap.add_argument("-reg", "--reg", type=int, default=0, help="Regularizaction")
    ap.add_argument("-dropout1", "--dropout1", type=float, default=0.0621, help="Dropout 1")
    ap.add_argument("-dropout2", "--dropout2", type=float, default=0.007, help="Dropout 2")
    ap.add_argument("-filters1", "--filters1", type=int, default=97, help="Filters")
    ap.add_argument("-filters2", "--filters2", type=int, default=39, help="Filters")
    ap.add_argument("-filters3", "--filters3", type=int, default=20, help="Filters")
    ap.add_argument("-filters4", "--filters4", type=int, default=11, help="Filters")
    ap.add_argument("-ts", "--ts_lenght", type=bool, default=2048, help="Time series lenght")
    ap.add_argument("-epochs", "--num_epoch", type=int, default=166, help="Epochs")
    ap.add_argument("-bs", "--batch_size", type=int, default=162, help="Batch size")

    
    args = vars(ap.parse_args())

    params = {
        'activation': args['activation'],
        'lr' : args['lr'], 
        'epochs': args['num_epoch'],
        'ts_lenght': args['ts_lenght'],
        'dropout1': args['dropout1'],
        'dropout2': args['dropout2'],
        'filters1': args['filters1'], 
        'filters2': args['filters2'],
        'filters3': args['filters3'],
        'filters4': args['filters4'],
        'batch_size': args['batch_size'],
    }
    
    with mlflow.start_run():

        model, val_loss, history = model_run(**params)
        model.save('models/cnn_reg/')
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        for step, (mloss, mvloss) in enumerate(zip(
                                                    history.history['loss'], history.history['val_loss']
                                                )):
            metrics = {'loss': float(mloss), 'val_loss': float(mloss)}
            mlflow.log_metrics(metrics, step=step)


            mlflow.keras.log_model(model, "class_model")

