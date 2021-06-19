from training_lstm import model_run
import optuna 
import numpy as np

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def objective(trial):

    params = {
        'file_name': 'D-SET(n=1000,ts_lenght=800,m1=30,m2=60).hkl',
        'activation': 'tanh',
        'lr' : 0.0001, 
        'reg' : trial.suggest_float("reg", 0.25, 0.5), 
        'dropout' : trial.suggest_float("dp1", 0, 1), 
        'num_units' :  trial.suggest_int("num_units", 300, 500),
        'epochs': trial.suggest_int("epochs", 100, 200),
        'batch_size':  trial.suggest_int("batch_size", 100, 200),
        'ts_lenght': 800,
        'm1':  30,
        'm2':  60,
    }

    model, val_loss, stats = model_run(**params)

    return 1 - np.mean(stats.history['mea']) #maksymalizacja błędu średnio bezwzględnego  


trials = 100
study = optuna.create_study(storage="sqlite:///hyper_lstm.db")
study.optimize(objective, n_trials=trials)