import numpy as np
import scipy.signal as signal
from scipy import interpolate
import sounddevice as sd
from copy import copy
import wave
import concurrent.futures as futures
import queue

from rtlsdr import RtlSdr 
from rtlsdr.helpers import *

sdr = RtlSdr()

# setup
sdr_sps = 2*256*256*16 # Hz
audio_sps = 44100 # Hz
fc = 94.9e6 # Hz - KUOW
dt = 1.0 / sdr_sps # seconds
sdr_nyquist = sdr_sps / 2.0

# SDR config
sdr.sample_rate = sdr_sps
sdr.center_freq = fc
sdr.gain = 42.0

# process SDR, can write that data to a shared buffer
# then playback callback can use same buffer, copying it to output
# might have sync clicks or something but should work ok

# global setting variables
SAMPLE_RATE = 44100
NYQUIST = SAMPLE_RATE / 2
BLOCK_SIZE = 2 ** 12
PITCH_SHIFT = 1

SAMPLE_BUS = queue.Queue()

# sig: input vector
# factor: amount to scale by, e.g. 2 = +1 octave, 0.5 = -1 octave.
def freq_scale(sig, factor = 2):
    x = np.arange(0, len(sig))
    f = interpolate.interp1d(x, (sig.imag, sig.real), bounds_error=False, fill_value=0)
    return f(x / factor)[0] + f(x / factor)[1]

# freq scale but on an array of stft blocks
def freq_scale_stft(stft, factor = 2):
    stft_s = np.transpose(stft)
    scaled = np.zeros_like(stft_s, dtype=complex)
    for i, ft in enumerate(stft_s):
        scaled[i] = freq_scale(ft, factor)
    return np.transpose(scaled)

# takes the stft of block, scales frequencies, and istft back to samples which are returned
# to use in callbacks etc.
def freq_scale_block(block, pitch_shift = 1, window='blackman', overlap=0.75):
    nop = int(BLOCK_SIZE * overlap)
    f, t, stft = signal.stft(block, SAMPLE_RATE, window=window, nperseg=BLOCK_SIZE, 
                             return_onesided=False, noverlap=nop)
    stft = freq_scale_stft(stft, pitch_shift)
    t, samples_back = signal.istft(stft, SAMPLE_RATE, window=window, nperseg=BLOCK_SIZE,
                                   input_onesided=False, noverlap=nop)
    return samples_back.real

def create_output_callback():
    print("making output")
    # write outdata to output
    def callback(outdata, frames, time, status):
        # get input from radio, perform DSP scale things, write output to outdata
        outdata[:, 0] = freq_scale_block(np.array([SAMPLE_BUS.get(block=True) for i in range(BLOCK_SIZE)]), PITCH_SHIFT)   
    return callback

# number, number, 6d tuple, 6d tuple, sound device -> (ndarray -> None)
# e.g. SDR sample rate, audio sample rate, SOS pre-filter, SOS post-filter, sound device -> buffer -> None
def create_process_callback(sdr_sps, audio_sps, presos, postsos):
    def callback(buffer, context):
        dt = 1.0 / sdr_sps # seconds
        sdr_nyquist = sdr_sps / 2.0

        # apply filter
        filt_samples = signal.sosfilt(presos, buffer)

        # find theta
        theta = np.arctan2(filt_samples.imag, filt_samples.real)

        # squelch
        abs_signal = np.abs(filt_samples)
        mean_abs_signal = np.mean(abs_signal)
        squelch_mask = abs_signal < (mean_abs_signal / 1.0)
        squelch_theta = copy(theta)
        np.putmask(squelch_theta, squelch_mask, 0.0)
        squelch_samples = copy(filt_samples)
        np.putmask(squelch_samples, squelch_mask, 0.0)

        # find dtheta
        dtheta_p0 = np.convolve([1, -1], squelch_theta, 'same')
        dtheta_pp = np.convolve([1, -1], (squelch_theta + np.pi) % (2 * np.pi), 'same')
        dtheta = np.where(abs(dtheta_p0) >= abs(dtheta_pp), dtheta_p0, dtheta_pp)

        # clean dtheta
        cdtheta = copy(dtheta)
        spikethresh = 0.5
        # there's got to be a faster way?
        for i in range(1,len(dtheta)-1):
            if (abs(dtheta[i])>spikethresh):
                cdtheta[i] = (dtheta[i-1]+dtheta[i+1]) / 2.0

        # downsample
        dsf = round(sdr_sps / audio_sps)
        audio_samples = signal.decimate(cdtheta, dsf, ftype='fir')

        # filter again
        filt_audio_samples = signal.sosfilt(postsos, audio_samples)
        
        # play
        # output is in filt_audio_samples
        for s in filt_audio_samples:
            SAMPLE_BUS.put(s, block=True)

        
    return callback

passband = 12000
stopband = sdr_nyquist - 1
presos = signal.iirdesign(passband, stopband, 0.1, 24, fs = sdr_sps, output = 'sos')
passband = [40, 8000]
stopband = [10, 10000]
postsos = signal.iirdesign(passband, stopband, 0.1, 48, fs = audio_sps, output = 'sos')

def read_sdr():
    sdr.read_samples_async(create_process_callback(sdr_sps, SAMPLE_RATE, presos, postsos), BLOCK_SIZE * np.ceil(sdr_sps / audio_sps))

def write_out():
    with sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, 
                         callback=create_output_callback(), finished_callback=write_out):
        sd.sleep(5000)

futures.ThreadPoolExecutor(max_workers=1).submit(read_sdr)

with sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=create_output_callback()):
    while True:
        response = input()
        if response in ('', 'Q', 'q'):
            break
        PITCH_SHIFT = float(response)

sdr.cancel_read_async()