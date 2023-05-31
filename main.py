import numpy as np
import queue
import concurrent.futures as futures

from consts import *
from processors import *
from sources import *
from dests import *

sample_bus = queue.Queue() # location to write source's samples to

processor = PitchPV(1) # object to use to manipulate the samples
processor.shift_factor = 1
# processor = Bypass()

source = SdrDecoder(sample_bus, SAMPLE_RATE) # set up the source
# source = WaveSource(sample_bus, "./resources/sc-vs-mc.wav")

dest = AudioOut(sample_bus, processor) # set up the sink

def start(source, callback, dest):
    source.start()
    print(sample_bus)
    dest.start(callback)

def mainloop():
    while True:
        response = input()
        if response in ('', 'Q', 'q'):
            break
        processor.shift_factor = float(response)

    source.stop()
    dest.stop()

start(source, mainloop, dest)