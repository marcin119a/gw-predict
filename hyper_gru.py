from training_lstm import model_run
import optuna 
import numpy as np

def objective(trial):
    dropout = trial.suggest_float("dp1", 0, 1)
    reg = trial.suggest_float("reg", 0.5, 0.25)
    num_units = trial.suggest_int("num_units", 8, 16)
    batch_size = trial.suggest_int("batch_size", 100, 200)
    epochs = trial.suggest_int("epochs", 100, 200)

    model, val_loss, stats = model_run(dropout=dropout, epochs=epochs,
                                       batch_size=batch_size, num_units=num_units, reg=reg)

    return 1 - np.mean(stats.history['loss']) #maksymalizacja 


trials = 100
study = optuna.create_study(storage="sqlite:///hyper_lstm.db", study_name="gru_study")
study.optimize(objective, n_trials=trials)