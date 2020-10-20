import math
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials

trials = MongoTrials('mongo://localhost:1234/optim/jobs', exp_key='exp1')
best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)