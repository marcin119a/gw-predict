from training_class import model_run
import optuna 
import numpy as np

def objective(trial):
    
    args = {
        'activation': 'tanh',
        'lr' : 0.001, 
        'epochs':  trial.suggest_int("epochs", 100, 200),
        'ts_lenght': 2048,
        'batch_size': trial.suggest_int("batch_size", 100, 200),
        'dropout1': trial.suggest_float("dp1", 0, 1),
        'dropout2': trial.suggest_float("dp2", 0, 1),
        'filters1': trial.suggest_int("filters1", 64, 128), 
        'filters2': trial.suggest_int("filters2", 32, 64),
        'filters3': trial.suggest_int("filters3", 16, 32),
        'filters4': trial.suggest_int("filters4", 8, 16),
    }

    model, val_loss, stats = model_run(**args)

    return 1 - np.mean(stats.history['accuracy']) #maksymalizacja 


trials = 100
study = optuna.create_study(storage="sqlite:///hyper_class.db")
study.optimize(objective, n_trials=trials)