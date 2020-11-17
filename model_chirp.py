from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras.optimizers as optim
from keras.regularizers import l1,l2
import numpy as np
from lstm_chirp import create_model, X_train, X_test, y_test, y_train
from datetime import datetime
import argparse



def model_run(X_train, y_test, activation='tanh', lr=9.345405211822168 * (10**-5), 
              reg=3.564391979952528 * (10** -5), dropout= 0.369863105580737, 
              num_layers=546, epoach=2, batch_size=2):
    
    model = create_model(activation=activation, lr=lr, reg=reg, dropout=dropout, num_layers=num_layers)
    stats = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoach, batch_size=batch_size)


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-activation", "--activation", type=str, default=False, help="Activation")
    ap.add_argument("-lr", "--lr", type=bool, default=False, help="Learning Rate")
    ap.add_argument("-reg", "--reg", type=bool, default=False, help="Regularizaction")
    ap.add_argument("-dropout", "--dropout", type=bool, default=False, help="Dropout")
    ap.add_argument("-num_layers", "--num_layers", type=bool, default=False, help="Num of layers")
    ap.add_argument("-epochs", "--num_epoch", type=bool, default=False, help="Epochs")
    ap.add_argument("-bs", "--batch_size", type=bool, default=False, help="Batch size")
    
    args = vars(ap.parse_args())
    activation = args['activation']
    lr = args['lr']
    reg = args['reg']
    dropout = args['dropout']
    num_layers = args['num_layers']

    params = {
        'X_train': X_train,
        'y_test': y_test,
        'activation': activation,
        'lr' : lr, 
        'reg' : reg, 
        'dropout' : dropout, 
        'num_layers' : num_layers
    }
    
    model_run(**params)