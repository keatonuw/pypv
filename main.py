import numpy as np
import queue

from consts import *
from processors import *
from sources import *
from dests import *

sample_bus = queue.Queue() # location to write source's samples to
processor = PitchPV(1) # object to use to manipulate the samples
# source = SdrDecoder(sample_bus, SAMPLE_RATE) # set up the source
source = WaveSource(sample_bus, "./resources/test_voice.wav")
dest = AudioOut(sample_bus, processor) # set up the sink

def start(source, processor, dest):
    source.start()
    dest.start()


start(source, processor, dest)

