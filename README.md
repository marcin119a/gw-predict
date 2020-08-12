Utilities for lstm model using gravitaion waves 

### Instalation: 




```bash
pip install -r requirements.txt
```

```bash
# We'll first download some data for this demonstration
!curl -O -J -L https://losc.ligo.org/s/events/LVT151012/H-H1_LOSC_4_V2-1128678884-32.gwf
```

Parameters for generating one waveform in multiple detectors
```python
from 
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

signal_h1, signal_l1, signal_v1 = generate_wave(params)
```
