""" Module includes utilities functions such as data generation,
plotting, etc.
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import regularizers
from keras.initializers import glorot_normal
from keras.optimizers import Adam

def generate_data(
    n=200,
    sig_ampl=1.0, sig_freq=0.1,
    noise_ampl=0.3, noise_freq=0.4
    ):
    """
    Generate mock time series with a sinusoidal signal and a sinusoidal noise
    source.

    Parameters:
    -----------
    n: int
        number of samples
    sig_ampl: float
        amplitude of the signal
    sig_freq: float
        frequnecy of the signal
    noise_ampl: float
        amplitude of the sinusoidal noise
    noise_freq: float
        frequnecy of the sinusoidal noise

    Returns:
    --------
    measured: numpy.ndarray
        measured signal
    noise: numpy.ndarray
        noise output from the witness channel
    clean: numpy.ndarray
        clean signal for testing purpose

    """
    t = np.arange(n)
    clean = sig_ampl * np.sin(sig_freq * np.pi * t)
    noise = noise_ampl * np.sin(noise_freq * np.pi * t)
    measured = clean + noise
    return measured, noise, clean

print(generate_data())

def create_model(input_shape, activation='tanh', lr=1e-3, reg=0.0, dropout=0.0):
    """
    Return a neural network model given the hyperparameters and input shape.
    """
    model = Sequential()
    model.add(LSTM(32,
                   input_shape          = input_shape,
                   activation           = activation,
                   kernel_regularizer   = regularizers.l2(reg),
                   kernel_initializer   = glorot_normal(),
                   bias_initializer     = 'ones',
                   dropout              = dropout,
                   name                 = 'What'))
    model.add(Dense(8,
                    activation          = activation,
                    kernel_regularizer  = regularizers.l2(reg),
                    kernel_initializer  = glorot_normal(),
                    name                = 'Is'))
    model.add(Dense(1,
                    kernel_regularizer  = regularizers.l2(reg),
                    kernel_initializer  = glorot_normal(),
                    name                = 'Love'))
    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

def set_plot_style():
    """
    provide matplotlib plotting style
    """
    plt.style.use('bmh')
    mpl.rcParams.update({
        'axes.grid': True,
        'axes.titlesize': 'medium',
        'font.family': 'serif',
        'font.size': 20,
        'grid.color': 'xkcd:grey',
        'grid.linestyle': '-',
        'grid.alpha': 0.5,
        'grid.linewidth': 1,
        'legend.borderpad': 0.2,
        'legend.fancybox': True,
        'legend.fontsize': 20,
        'legend.framealpha': 0.7,
        'legend.handletextpad': 0.1,
        'legend.labelspacing': 0.2,
        'legend.loc': 'best',
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{txfonts}'
    })

    mpl.rc("savefig", dpi=100)
    mpl.rc("figure", figsize=(8, 5))
