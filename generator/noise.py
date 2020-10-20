from generator.generator_waveform import generate_wave
import pycbc.psd
import pycbc.noise

apx = 'SEOBNRv4'


params = {
    'approximant':apx,
    'mass1':100,
    'mass2':10,
    'spin1z':0.9,
    'spin2z':0.4,
    'inclination':1.23,
    'coa_phase':2.45,
    'delta_t':1.0/4096,
    'f_lower':40
}

hp, hc, signal_v1 = generate_wave(params)


flow = 90.0
delta_f = 1.0 / 16
flen = int(2048 / delta_f) + 1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
delta_t = 1.0 / 4096
seconds = float(len(hp)) / 4096
print('Seconds: '+str(seconds))
tsamples = int(seconds / delta_t)
ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127)
result = [x + y for x,y in zip(ts, hp)]