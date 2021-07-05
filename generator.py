import numpy as np
from genwave.waveform import generate_wave
import tensorflow as tf
from sklearn.model_selection import train_test_split

def chrip_mass(m1, m2):
  return pow(m1*m2,3/5)/ pow(m1 + m2, 1/5)


def random_dataset(m1, m2, n_steps, batch_size, channels):
  apx = 'SEOBNRv4'
  y = np.zeros((batch_size, 1))
  X = np.zeros((batch_size, n_steps, channels))
  for i in range(batch_size):
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
      signal_h1.resize(n_steps) #crop signal into window (0, n_steps)
      signal_l1.resize(n_steps) #crop signal into window (0, n_steps)
      signal_v1.resize(n_steps) #crop signal into window (0, n_steps)

      signal_h1_np = np.array([step for step in signal_h1])    
      signal_l1_np = np.array([step for step in signal_l1])    
      signal_v1_np = np.array([step for step in signal_v1])    

      # from ex. 10^{-20} amplitude into (0,1)
      scale = 10**(20)
      #X[i,:,0] = signal_h1_np * scale
      #X[i,:,1] = signal_l1_np * scale
      #X[i,:,2] = signal_v1_np * scale
      X[i,:,0] = tf.keras.utils.normalize(signal_h1_np, axis = -1)  
      X[i,:,1] = tf.keras.utils.normalize(signal_l1_np, axis = -1)  
      X[i,:,2] = tf.keras.utils.normalize(signal_v1_np, axis = -1)  
      
      
      #without normalizaction
      y[i,0] = chrip_mass(mass1, mass2)
     
        
  return X, y


def split_dataset(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  return (X_train, X_test, y_train, y_test)


from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector


def generate_wave(params):


  hp, hc = get_td_waveform(**params)

  det_h1 = Detector('H1')
  det_l1 = Detector('L1')
  det_v1 = Detector('V1')

  # Choose a GPS end time, sky location, and polarization phase for the merger
  # NOTE: Right ascension and polarization phase runs from 0 to 2pi
  #       Declination runs from pi/2. to -pi/2 with the poles at pi/2. and -pi/2.
  end_time = 1192529720
  declination = 0.65
  right_ascension = 4.67
  polarization = 2.34
  hp.start_time += end_time
  hc.start_time += end_time

  signal_h1 = det_h1.project_wave(hp, hc,  right_ascension, declination, polarization)
  signal_l1 = det_l1.project_wave(hp, hc,  right_ascension, declination, polarization)
  signal_v1 = det_v1.project_wave(hp, hc,  right_ascension, declination, polarization)

  return signal_h1, signal_l1, signal_v1
