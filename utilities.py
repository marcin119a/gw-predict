from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scaler(dataset):
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset.reshape(-1,1))
  
  return dataset


def split_dataset(dataset):
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  
  return train, test


def preprocessing(dataset):
  dataset = np.array([i for i in dataset])
  dataset = scaler(dataset)
