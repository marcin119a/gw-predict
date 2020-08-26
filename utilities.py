import numpy as np


def split_dataset(dataset):
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  
  return train, test


def preprocessing(dataset):
  dataset = np.array([i for i in dataset])
  dataset = scaler(dataset)
  
  return dataset

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def split_sequences_wave(sequences, params):
  X, y = list(), list()
  for seq in sequences:
    X.append([seq])
  
  for x in params:
  	y.append(params)
  
  return array(X), y
