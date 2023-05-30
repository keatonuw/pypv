import sounddevice as sd
import numpy as np

from consts import *

'''
This file contains the destination objects
'''

class AudioOut:

    def __init__(self, source_queue, processor):
        self.processor = processor
        self.source_queue = source_queue

    def create_output_callback(self):
        # write outdata to output
        def callback(outdata, frames, time, status):
            # get input from radio, perform DSP scale things, write output to outdata
            outdata[:, 0] = self.processor.process(np.array([self.source_queue.get(block=True) for i in range(BLOCK_SIZE)]))   
        return callback

    def start(self):
        with sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=self.create_output_callback()):
            while True:
                response = input()
                if response in ('', 'Q', 'q'):
                    break