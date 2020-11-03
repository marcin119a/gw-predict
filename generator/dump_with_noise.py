from generator_waveform import generate_wave
import pycbc.psd
import pycbc.noise
import numpy as np 
import argparse
from utilities import mass_quarter

apx = 'SEOBNRv4'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=100, help="the number of signals")
    ap.add_argument("-m1", "--mass1", type=int, default=10, help="First mass of black hole")
    ap.add_argument("-m2", "--mass2", type=int, default=40, help="Second mass of black hole")
    ap.add_argument("-time_steps", "--ts", type=int, default=1400, help="Time steps for single signal")
    ap.add_argument("-quark", "--qr", type=bool, default=False, help="Quark parameters")

    args = vars(ap.parse_args())

    m1 = args['mass1']
    m2 = args['mass2']
    iteraction = args['n']
    n_steps = args['ts']
    
    X 	 = []
    y 	 = []

    for x in range(iteraction):
        apx = 'SEOBNRv4'

        mass1 = np.random.uniform(m1, m2)
        mass2 = np.random.uniform(m1, m2)

        params = {
            'approximant':apx,
            'mass1':mass1,
            'mass2':mass2,
            'spin1z':0.9,
            'spin2z':0.4,
            'inclination':1.23,
            'coa_phase':2.45,
            'delta_t':1.0/4096,
            'f_lower':40
        }

        hp, hc, signal_v1 = generate_wave(params)
        hp.resize(n_steps)

        flow = 90.0
        delta_f = 1.0 / 16
        flen = int(2048 / delta_f) + 1
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
        delta_t = 1.0 / 4096
        seconds = float(len(hp)) / 4096

        tsamples = int(seconds / delta_t)
        ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127)
        result = [x + y for x,y in zip(ts, hp)]
        y1 = mass_quarter(mass1, mass2)
        X.append(result)
        y.append(y1)
    return X, y


if __name__ == '__main__':
    main()

