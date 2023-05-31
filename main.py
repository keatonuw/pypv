import queue
import sys

from consts import *
from processors import *
from sources import *
from dests import *

def main(sample_bus: queue.Queue, processor: Processor, source: Source, dest: AudioOut):
    def start(source, callback, dest):
        source.start()
        dest.start(callback)

    def mainloop():
        while True:
            response = input()
            if response in ('', 'Q', 'q'):
                break
            processor.control(float(response))

        source.stop()
        dest.stop()

    start(source, mainloop, dest)
    exit(0)

def usage():
    print("USAGE: python main.py (-s (<filename>|sdr (<freq>)?|in) -p (pitch|bypass))?")
    exit(1)

if __name__ == "__main__":
    # -s (<filename>|sdr|in)
    # -p (pitch|bypass)
    sample_bus = queue.Queue() # location to write source's samples to

    # set up processor
    if "-p" in sys.argv:
        idx = sys.argv.index("-p") + 1
        if sys.argv[idx] == "pitch":
            processor = PitchPV(1)
        elif sys.argv[idx] == "bypass":
            processor = Bypass()
        else:
            usage()
    else:
        processor = PitchPV(1)

    # set up source
    if "-s" in sys.argv:
        idx = sys.argv.index("-s") + 1
        if sys.argv[idx] == "sdr":
            try:
                fc = float(sys.argv[idx + 1])
            except:
                fc = 94.9
            source = SdrFIRDemod(sample_bus, SAMPLE_RATE, fc)
        elif sys.argv[idx] == "in":
            print("RT Input not implemented")
            usage()
        else:
            source = WaveSource(sample_bus, sys.argv[idx])
    else:
        source = SdrFIRDemod(sample_bus, SAMPLE_RATE)

    # set up dest
    dest = AudioOut(sample_bus, processor) # set up the sink

    print("Starting PYPV. Enter control amounts or 'Q' to quit.")
    main(sample_bus, processor, source, dest)
