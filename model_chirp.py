from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras.optimizers as optim
from keras.regularizers import l1,l2
import numpy as np
from lstm_chirp import create_model
from datetime import datetime
import argparse
import tensorflow as tf
import hickle as hkl
import mlflow

def model_run(file_name, activation='tanh', lr=9.345405211822168 * (10**-5),
             reg=3.564391979952528 * (10** -5), dropout=0, num_layers=500,
             epochs=10, batch_size=2):
    
    model = create_model(activation=activation, lr=lr, reg=reg, dropout=dropout, num_layers=num_layers)

    array_hkl = hkl.load(file_name)
    X_train = array_hkl.get('xtrain')
    X_test = array_hkl.get('xtest')
    y_train = array_hkl.get('ytrain')
    y_test = array_hkl.get('ytest')

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], 1)
    X_test =  X_test.reshape(X_test.shape[0], X_test.shape[2], 1)

    y_train = tf.keras.utils.normalize(y_train)  
    y_test = tf.keras.utils.normalize(y_test)  

    y_test = np.squeeze(y_test)
    y_train = np.squeeze(y_train)

    stats = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)


    y_pred = model.predict(X_test)
    date = datetime.today().strftime("%Y-%m-%d")
    filename = f'model/model-chirp{date}.h5'
    model.save(filename, save_format='h5')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(stats.history['loss'], label='train')
    ax.plot(stats.history['val_loss'], label='test')
    ax.set(xlabel='Epochs', ylabel='MSE Loss', xticks=np.arange(0, 100, 5))
    ax.legend()
    fig.tight_layout()

    fig.savefig('loss_vs_val_chirp{0}.png'.forma(date))

    y_test = np.squeeze(y_test)
    y_pred = np.squeeze(y_pred)

    print('RMSE: {}'.format(np.sqrt(np.mean((y_test - y_pred)**2))))

    return model, np.sqrt(np.mean((y_test - y_pred)**2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", type=str, default='data/D-SET(100,1200)-chirp.hkl', help="File")
    ap.add_argument("-act", "--activation", type=str, default='tanh', help="Activation")
    ap.add_argument("-lr", "--lr", type=bool, default=0, help="Learning Rate")
    ap.add_argument("-reg", "--reg", type=bool, default=0, help="Regularizaction")
    ap.add_argument("-dropout", "--dropout", type=bool, default=0.37, help="Dropout")
    ap.add_argument("-nn", "--num_layers", type=bool, default=546, help="Num of neurons")
    ap.add_argument("-epochs", "--num_epoch", type=bool, default=100, help="Epochs")
    ap.add_argument("-bs", "--batch_size", type=bool, default=100, help="Batch size")
    
    args = vars(ap.parse_args())

    params = {
        'file_name': args['file'],
        'activation': args['activation'],
        'lr' : args['lr'], 
        'reg' : args['reg'], 
        'dropout' : args['dropout'], 
        'num_layers' :  args['num_layers'],
        'epochs': args['num_epoch'],
        'batch_size':  args['batch_size']
    }
    
    with mlflow.start_run():

        model, rmse = model_run(**params)

        mlflow.log_param("file", args['file'])
        mlflow.log_metric("rmse", rmse)

        mlflow.keras.save_model(model, "model")

