import tensorflow as tf
from generator_waveform import generate_wave
from hyperopt import hp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def random_dataset(m1, m2):
  apx = 'SEOBNRv4'
  

  X = []
  y = []
  for _ in range(100):
      mass1 = np.random.uniform(m1,m2)
      mass2 = np.random.uniform(m1,m2)
      scaler = MinMaxScaler(feature_range=(0, 1))
      
      params = {
          'approximant': apx,
          'mass1': mass1,
          'mass2': mass2,
          'spin1z': 0.9,
          'spin2z': 0.4,
          'inclination': 1.23,
          'coa_phase': 2.45,
          'delta_t': 1.0/4096,
          'f_lower': 40
      }

      signal_h1, signal_l1, signal_v1 = generate_wave(params)
      signal_h1.resize(1400)
      x1 = np.array([step for step in signal_h1])
      x1 = scaler.fit_transform(x1.reshape(-1,1))

      X.append(x1)
      y.append(np.array([mass1, mass2]))


  return np.array(X), np.array(y)
