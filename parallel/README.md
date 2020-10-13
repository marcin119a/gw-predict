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

### Run:
```
    setpid ./mongo.sh
    setpid ./worker.sh
    
```