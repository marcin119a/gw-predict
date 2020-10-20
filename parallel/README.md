### Parallel setup: 

#### Install mongodb
```bash
    conda install mongodb
```

#### Run monodb
```bash
    mongod --dbpath . --port 1234
```

#### Run worker
```bash
    hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1
```

#### Run optymalioptimizationzaction
```
    python parallel.py
```


#### Requirements

```bash
    pymongo
    hyperopt
```

### Run tmux session:
```bash
    tmux new-session
    sh mongo.db
```
#### Detach from session:
```
ctrl + b d
```

```bash
    tmux new-session
    sh worker.sh
```

#### Show all sessions:
```
Ctrl + b s
```


```
    setpid ./mongo.sh
    setpid ./worker.sh
    
```

Resources:
* https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB
* https://tmuxcheatsheet.com/