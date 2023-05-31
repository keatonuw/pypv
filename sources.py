import numpy as np
import copy
import scipy.signal as signal
import concurrent.futures as futures
import wave
import queue

from consts import *

from rtlsdr import RtlSdr

class Source:
    def start(self):
        pass

    def stop(self):
        pass

class SdrDecoder(Source):

    def __init__(self, dest_queue, sample_rate=44100,
                 presos=signal.iirdesign(8000, SDR_SPS / 2 - 1, 0.1, 24, fs = SDR_SPS, output = 'sos'), 
                 postsos=signal.iirdesign([50, 10000], [20, 12000], 0.1, 48, fs = 44100, output = 'sos')):
        self.sdr = RtlSdr()
        self.audio_sps = sample_rate
        self.sdr_sps = SDR_SPS # Hz
        self.fc = 94.9e6 # Hz - KUOW
        self.dt = 1.0 / self.sdr_sps # seconds
        self.sdr_nyquist = self.sdr_sps / 2.0
        self.sdr.sample_rate = self.sdr_sps
        self.sdr.center_freq = self.fc
        self.sdr.gain = 42.0
        self.dest_queue = dest_queue
        self.presos = presos
        self.postsos = postsos

    # create a callback for use with SDR
    def __getCallback(self):
        def callback(buffer, context):
            # filter
            filt_samples = signal.sosfilt(self.presos, buffer)

            # get theta
            theta = np.arctan2(filt_samples.imag, filt_samples.real)

            # squelch
            abs_signal = np.abs(filt_samples)
            mean_abs_signal = np.mean(abs_signal)
            squelch_mask = abs_signal < (mean_abs_signal / 1.0)
            squelch_theta = copy.copy(theta)
            np.putmask(squelch_theta, squelch_mask, 0.0)
            squelch_samples = copy.copy(filt_samples)
            np.putmask(squelch_samples, squelch_mask, 0.0)

            # find dtheta
            dtheta_p0 = np.convolve([1, -1], squelch_theta, 'same')
            dtheta_pp = np.convolve([1, -1], (squelch_theta + np.pi) % (2 * np.pi), 'same')
            dtheta = np.where(abs(dtheta_p0) >= abs(dtheta_pp), dtheta_p0, dtheta_pp)

            # clean dtheta
            cdtheta = copy.copy(dtheta)
            spikethresh = 0.5
            # there's got to be a faster way?
            for i in range(1,len(dtheta)-1):
                if (abs(dtheta[i])>spikethresh):
                    cdtheta[i] = (dtheta[i-1]+dtheta[i+1]) / 2.0

            # downsample
            dsf = round(self.sdr_sps / self.audio_sps)
            audio_samples = signal.decimate(cdtheta, dsf, ftype='fir')

            # filter again
            filt_audio_samples = signal.sosfilt(self.postsos, audio_samples)
        
            # play
            # output is in filt_audio_samples
            for s in filt_audio_samples:
                self.dest_queue.put(s, block=True)
        
        return callback

    # start writing to the queue!
    def start(self):
        block_size = BLOCK_SIZE * np.ceil(self.sdr_sps / self.audio_sps)
        def read_sdr():
            self.sdr.read_samples_async(self.__getCallback(), block_size)
        futures.ThreadPoolExecutor(max_workers=1).submit(read_sdr)

    def stop(self):
        self.sdr.cancel_read_async()

# like the older SDRDecoder above, but using a FIR filter for pre- and post- filtering.
class SdrFIRDemod(Source):

    def __init__(self, dest_queue, sample_rate=44100, tuning=94.9,
                 pre_lp=70000, pre_kernel_size=128,
                 post_low_cut=20, post_hi_cut=20000, post_kernel_size=128):
        self.sdr = RtlSdr()
        self.audio_sps = sample_rate
        self.sdr_sps = SDR_SPS # Hz
        self.fc = tuning * 1e6 # KEXP
        self.dt = 1.0 / self.sdr_sps # seconds
        self.sdr_nyquist = self.sdr_sps / 2.0
        self.sdr.sample_rate = self.sdr_sps
        self.sdr.center_freq = self.fc
        self.sdr.gain = 42.0
        self.dest_queue = dest_queue

        fac = (2 * pre_lp / SDR_SPS)
        self.pre_kernel =  fac * np.sinc(fac * (np.arange(0, pre_kernel_size) - (pre_kernel_size - 1) / 2))
        self.pre_kernel = self.pre_kernel * np.hanning(pre_kernel_size)

        fac = (2 * post_hi_cut / SAMPLE_RATE)
        self.post_kernel = fac * np.sinc(fac * (np.arange(0, post_kernel_size) - (post_kernel_size - 1) / 2))
        fac = (2 * post_low_cut / SAMPLE_RATE)
        self.post_kernel -= fac * np.sinc(fac * (np.arange(0, post_kernel_size) - (post_kernel_size - 1) / 2))
        self.post_kernel = self.post_kernel * np.hanning(post_kernel_size)
        # self.postsos = postsos

    # create a callback for use with SDR
    def __getCallback(self):
        def callback(buffer, context):
            # filter
            filt_samples = np.convolve(buffer, self.pre_kernel, 'same')
            # signal.sosfilt(self.presos, buffer)

            # get theta
            theta = np.arctan2(filt_samples.imag, filt_samples.real)

            # squelch
            abs_signal = np.abs(filt_samples)
            mean_abs_signal = np.mean(abs_signal)
            squelch_mask = abs_signal < (mean_abs_signal / 1.0)
            squelch_theta = copy.copy(theta)
            np.putmask(squelch_theta, squelch_mask, 0.0)
            squelch_samples = copy.copy(filt_samples)
            np.putmask(squelch_samples, squelch_mask, 0.0)

            # find dtheta
            dtheta_p0 = np.convolve([1, -1], squelch_theta, 'same')
            dtheta_pp = np.convolve([1, -1], (squelch_theta + np.pi) % (2 * np.pi), 'same')
            dtheta = np.where(abs(dtheta_p0) >= abs(dtheta_pp), dtheta_p0, dtheta_pp)

            # clean dtheta
            cdtheta = copy.copy(dtheta)
            spikethresh = 0.5
            # there's got to be a faster way?
            for i in range(1,len(dtheta)-1):
                if (abs(dtheta[i])>spikethresh):
                    cdtheta[i] = (dtheta[i-1]+dtheta[i+1]) / 2.0

            # downsample
            dsf = round(self.sdr_sps / self.audio_sps)
            audio_samples = signal.decimate(cdtheta, dsf, ftype='fir')

            # filter again
            filt_audio_samples = np.convolve(audio_samples, self.post_kernel, 'same')
        
            # play
            # output is in filt_audio_samples
            for s in filt_audio_samples:
                self.dest_queue.put(s, block=True)
        
        return callback

    # start writing to the queue!
    def start(self):
        block_size = BLOCK_SIZE * np.ceil(self.sdr_sps / self.audio_sps)
        def read_sdr():
            self.sdr.read_samples_async(self.__getCallback(), block_size)
        futures.ThreadPoolExecutor(max_workers=1).submit(read_sdr)

    def stop(self):
        self.sdr.cancel_read_async()


class WaveSource(Source):

    def __init__(self, dest_queue: queue.Queue, file):
        self.dest_queue = dest_queue
        self.file = file

    def start(self):
        with wave.open(self.file, 'r') as w:
            print(w.getparams())
            buf = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            buf = buf.astype(dtype=np.float32)
            for s in buf / 2 ** 15:
                self.dest_queue.put(s, block=True)