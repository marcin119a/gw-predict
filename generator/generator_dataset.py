import numpy as np
from generator_waveform import generate_wave
import tensorflow as tf
from utilities import mass_quarter, max_ts

def random_dataset(m1, m2, n_steps, iteraction, chirp=True, max_model=False):
  apx = 'SEOBNRv4'
  

  X 	 = []
  X_norm = []
  y 	 = []
  y_norm = []

  for _ in range(iteraction):
      mass1 = np.random.uniform(m1, m2)
      mass2 = np.random.uniform(m1, m2)
      
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
      signal_h1.resize(n_steps)

      x1 = np.array([step for step in signal_h1])    
      x1_norm = tf.keras.utils.normalize(x1, axis = -1)  
      
      if chirp == True:
        y1 = mass_quarter(mass1, mass2)
        #without normalizaction
        y_norm.append(y1)
        y.append(y1)

      else:
        if max_model== True:
          y1 = max(x1_norm) 
          y_norm.append(y1)
          y.append(y1)
          
        else:
          y1 = np.array([mass1, mass2])
          y.append(y1)

          y1_norm = tf.keras.utils.normalize(y1, axis = -1)
          y_norm.append(y1_norm)

      X.append(x1)
      X_norm.append(x1_norm)
      



  return np.array(X_norm), np.array(y_norm), np.array(X), np.array(y)