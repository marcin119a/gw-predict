from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scaler(dataset):
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset.reshape(-1,1))
return dataset
