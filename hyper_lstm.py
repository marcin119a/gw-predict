from hyperopt.mongoexp import MongoTrials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import keras.optimizers as optim
import numpy as np
import tensorflow as tf 
from modle_gru import model_run
import argparse
import mlflow

ap = argparse.ArgumentParser()
ap.add_argument("-parallel", "--pr", type=bool, default=False, help="Parallel computation")
ap.add_argument("-batch_norm", "--bn", type=bool, default=False, help="Batch normalizaction")

args = vars(ap.parse_args())


# define a search space

space = {
    'activation':hp.choice('activation', ('relu', 'tanh')),
    'lr': hp.loguniform('lr', np.log(1e-6), np.log(1e-2)), 
    'dropout':hp.uniform('dropout', 0.0, 1.0), 
    'reg':hp.uniform('reg', 1e-6, 1e-3), 
    'num_layers': hp.uniformint('num_layers', 64, 1024),
    'file_name': 'D-SET(n=1000,time_steps=800).hkl',
    'epochs': 100,
    'batch_size':  100,
    'bn': True
}

# define loss function
def loss(params):
    model, val_loss, rmse, history = model_run(**params)
    print('Loss: {}'.format(val_loss))
    
    with mlflow.start_run():

        model, val_loss, rmse, history = model_run(**params)
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        for step, (mloss, mvloss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            metrics = {'loss': float(mloss), 'val_loss': float(mvloss)}
            mlflow.log_metrics(metrics, step=step)

        
        mlflow.log_metric("rmse", rmse)

        mlflow.keras.log_model(model, "lstm-pycbc")

    return {'loss': val_loss, 'status': STATUS_OK}

if __name__ == "__main__":
    print('Begin tuning')
    print('------------')

    if args["pr"]:
        trials = MongoTrials('mongo://localhost:1234/optim/jobs', exp_key='exp1')
    else:
        trials = Trials()

    best_params = fmin(loss,
                    space = space,
                    algo = tpe.suggest,
                    max_evals = 20,
                    trials = trials)
    print('')
    print('Best parameters:')
    print('----------------')
    best_params['activation'] = ['relu', 'tanh'][best_params['activation']]
    for k, v in best_params.items():
        print('{} = {}'.format(k, v))