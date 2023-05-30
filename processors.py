import scipy.signal as signal
import scipy.interpolate as interpolate
import numpy as np
from consts import *

'''
This file contains the processor objects used to transform incoming blocks into
new blocks.
'''

PITCH_MODE = 'pitch'
TIME_MODE = 'time'

# interpolate a STFT.
# sig: input vector.
# factor: amount to scale by, e.g. 2 = +1 octave, 0.5 = -1 octave.
def scale(sig, factor = 2):
    x = np.arange(0, len(sig))
    f = interpolate.interp1d(x, (sig.imag, sig.real), bounds_error=False, fill_value=0)
    return f(x / factor)[0] + f(x / factor)[1]

# scale, but on an array of stft blocks.
# stft: input array of STFTs.
# factor: amount to scale by.
# transpose: whether to transpose the matrix. True results in pitch shift, False in time scale.
def scale_stft(stft, factor = 2, transpose=True):
    if transpose:
        stft_s = np.transpose(stft)
    scaled = np.zeros_like(stft_s, dtype=complex)
    for i, ft in enumerate(stft_s):
        scaled[i] = scale(ft, factor)
    if transpose:
        return np.transpose(scaled)
    return scaled

# handles scaling a block in the time domain
# block: block of samples to process.
# pitch_shift: amount to scale by.
# window: window to use for the STFT.
# overlap: amount of window overlap for the STFTs.
# mode: either 'pitch' or 'time' for pitch shifting or time stretching.
def scale_block(block, pitch_shift = 1, window='blackman', overlap=0.75, mode=PITCH_MODE):
    nop = int(BLOCK_SIZE * overlap)
    f, t, stft = signal.stft(block, SAMPLE_RATE, window=window, nperseg=BLOCK_SIZE, 
                            return_onesided=False, noverlap=nop)
    stft = scale_stft(stft, pitch_shift, mode == PITCH_MODE)
    t, samples_back = signal.istft(stft, SAMPLE_RATE, window=window, nperseg=BLOCK_SIZE,
                                input_onesided=False, noverlap=nop)
    return samples_back.real

class PitchPV:

    def __init__(self, shift_factor = 1):
        self.shift_factor = shift_factor

    # call this to process a block with the settings of this object
    def process(self, block):
        return scale_block(block, self.shift_factor, mode='pitch')
    
class TimePV:

    def __init__(self, stretch_factor = 1):
        self.stretch_factor = stretch_factor

    def process(self, block):
        return scale_block(block, self.stretch_factor, mode='time')

