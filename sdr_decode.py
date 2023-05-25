import numpy as np
import copy
import scipy.signal as signal

# number, number, 6d tuple, 6d tuple, sound device -> (ndarray -> None)
# e.g. SDR sample rate, audio sample rate, SOS pre-filter, SOS post-filter, sound device -> buffer -> None
def create_process_callback(sdr_sps, audio_sps, presos, postsos, dest_queue):
    print("making sdr")
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
            dest_queue.put(s, block=True)

        
    return callback