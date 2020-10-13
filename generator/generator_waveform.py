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