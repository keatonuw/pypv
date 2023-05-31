import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.interpolate import CubicSpline
import numpy as np
from consts import *

'''
This file contains the processor objects used to transform incoming blocks into
new blocks.

It also contains routines used by these objects.

Currently only pitch shifting is fully implemented.
'''

PITCH_MODE = 'pitch'
TIME_MODE = 'time'

# interpolate a STFT.
# sig: input vector.
# factor: amount to scale by, e.g. 2 = +1 octave, 0.5 = -1 octave.
def scale(sig, factor = 2):
    x = np.arange(0, len(sig))
    f = interpolate.PchipInterpolator(x, sig, extrapolate=True)
    # f = interpolate.CubicSpline(x, sig, bc_type='clamped')
    return f(x / factor) * max(factor, 1 / factor) #[0] + f(x / factor)[1]

# scale, but on an array of stft blocks.
# stft: input array of STFTs.
# factor: amount to scale by.
# transpose: whether to transpose the matrix. True results in pitch shift, False in time scale.
def scale_stft(stft, factor = 2, transpose=True):
    if transpose:
        stft = np.transpose(stft)
    scaled = np.zeros_like(stft, dtype=complex)
    for i, ft in enumerate(stft):
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
def scale_block(block, pitch_shift = 1, window='hann', overlap=0.75, mode=PITCH_MODE):
    nop = int(FFT_BLOCK_SIZE * overlap)
    f, t, stft = signal.stft(block, SAMPLE_RATE, window=window, nperseg=FFT_BLOCK_SIZE, 
                            return_onesided=True, noverlap=nop)
    stft = scale_stft(stft, pitch_shift, mode == PITCH_MODE)
    t, samples_back = signal.istft(stft, SAMPLE_RATE, window=window, nperseg=FFT_BLOCK_SIZE,
                                input_onesided=True, noverlap=nop)
    samples_back = samples_back[len(samples_back) // 2 - BLOCK_SIZE // 2:len(samples_back) // 2 + BLOCK_SIZE // 2]
    return samples_back.real

class Processor:
    def process(self, block):
        pass

    def control(self, value):
        pass

class PitchPV(Processor):

    def __init__(self, shift_factor = 1, pre_lp = 20000):
        self.shift_factor = shift_factor
        self.prev_half = np.zeros(BLOCK_SIZE)
        self.hann = np.hanning(BLOCK_SIZE)

        fac = (2 * pre_lp / SAMPLE_RATE)
        kernel_size = 128

        # Zolzer sinc FIR LP
        self.prefir = fac * np.sinc(fac * (np.arange(0, kernel_size) - (kernel_size - 1) / 2))

    # call this to process a block with the settings of this object
    def process(self, block):
        scaled = scale_block(np.convolve(np.append(self.prev_half, block), self.prefir, 'same'), self.shift_factor, mode='pitch')
        self.prev_half = block
        return scaled
    
    def control(self, value):
        self.shift_factor = max(value, 0.1)
    
class TimePV(Processor):

    def __init__(self, stretch_factor = 1):
        self.stretch_factor = stretch_factor

    def process(self, block):
        return scale_block(block, self.stretch_factor, mode='time')
    
    def control(self, value):
        self.stretch_factor = max(value, 0.1)
    
class Bypass(Processor):

    def process(self, block):
        return block
    
    def control(self, value):
        pass

