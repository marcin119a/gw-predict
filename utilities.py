import hickle as hkl
import h5py
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split #added
import random


def split_dataset(file_name):
  array_hkl = hkl.load(file_name)
  X_train = array_hkl.get('xtrain')
  X_test = array_hkl.get('xtest')
  y_train = array_hkl.get('ytrain')
  y_test = array_hkl.get('ytest')

  return X_test, X_train, y_test, y_train

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_data_assign_labels_one_file(filename, idx, filtering=False, fmin=0, fmax=0, snr_tr=0): 
    
    """
    This function reads one hdf5 file and associate a class idx, if snr_th is given a threshold in snr is applied to the signal class
    Args: 
        label_list 
    Returns: 
        segments: array of time series
        labels: indicies of labels from label_list 
    
    """

    segments = [] 
    labels = [] 
    metadata = []
    fs = 2048   
    dur = 1
    if(filtering): print("Filtering with fmin",fmin,"fmax",fmax)
    with h5py.File(filename, "r") as f:

            try: 
                groups = f.keys()

                for key in f.keys():

                    grp = f[key]
                                        
                    if ((grp['time series'][:]).size == (fs*dur)):

                            data = grp['time series'][:]
                            if(filtering): data = butter_bandpass_filter(grp['time series'][:], fmin, fmax, fs, order=5)
                            segments.append(data)
                            labels.append(idx)

                            #I create a fake vector of metadata
                            #made of 17 zeroes because this is the duration
                            #of the metadata in the signal
                            addition = np.zeros(17)
                            metadata.append(addition)


            except IOError:         
                    print("Problem with {} file.".format(f.filename))

    f.close()

    x = segments
    y = list(zip(labels,metadata)) 

    return x,y


def split(file_name, file_signal, file_noise, file_glitch):
# Three classes: noise, noise+signal, glitch  
    LABELS = ["noise", "signal", "glitches"]

    
    x0, y0 = read_data_assign_labels_one_file(file_name+file_noise, 0, True, 20, 1000)
    x1, y1 = read_data_assign_labels_one_file(file_name+file_signal, 1, True, 20, 1000)        

    #split train and test
    x0_train, x0_test, y0_train, y0_test = train_test_split(x0, y0, test_size=0.5, random_state=24)
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.5, random_state=24)
    
    x00_train = [x0_train, x1_train]
    x00_test = [x0_test, x1_test]
    y00_train = [y0_train, y1_train]
    y00_test = [y0_test, y1_test]

    x2, y2 = read_data_assign_labels_one_file(file_name+file_glitch,2,True,20,1000)
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.5, random_state=24)
        
    x00_train.append(x2_train)
    x00_test.append(x2_test)
    y00_train.append(y2_train)
    y00_test.append(y2_test)
    


    #I create the training and testing sets combining the training and testing vector of the three classes
    x_train = sum(x00_train, [])
    x_test = sum(x00_test, [])
    yy_train = sum(y00_train, [])
    yy_test = sum(y00_test, [])

    #I shuffle the element of the training and testing sets
    combined_train = list(zip(x_train, yy_train ))    
    random.shuffle(combined_train)
    x_train, yy_train = zip(*combined_train)

    combined_test = list(zip(x_test, yy_test ))    
    random.shuffle(combined_test)
    x_test, yy_test = zip(*combined_test)

    #I divide my y from the metadate vector
    y_train, meta_train = zip(*yy_train)
    y_test, meta_test = zip(*yy_test)


    y_train = np.asarray(y_train)
    x_train = np.asarray(x_train)
    y_test = np.asarray(y_test)
    x_test = np.asarray(x_test)

    num_detectors = 1 
    num_classes = len(LABELS)
    num_time_periods = x_train.shape[1]
    input_shape = num_time_periods*num_detectors
    x_train = x_train.reshape(x_train.shape[0], input_shape)

    # Convert type for Keras 
    x_train = x_train.astype("float32")
    y_train = y_train.astype("float32")


    return x_test, y_test, x_train, y_train, num_detectors, num_classes, input_shape, LABELS


