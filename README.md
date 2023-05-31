# pypv
Python implementation of a Phase Vocoder for real-time CLI use with a RTLSDR (wav files are also supported!). Currently meant for pitch shifting incoming signals.

## Requirements
pypv requires a few additional python packages:
* numpy
* scipy
* rtlsdr

## Usage
After cloning the repository, the simplest way to get going is to plug in a RTLSDR and run:
```
python pypv.py
```
However, additional flags are available. `-s` allows source selection. Sources can be either a wav file or the SDR, with specifiable tuning. For example:
```
python pypv.py -s sdr 90.4
```
> Sets source to SDR with a center frequency of 90.4MHz

```
python pypv.py -s ./resources/test_chirp.wav
```
> Sets source to the contents of test_chirp.wav

The `-p` flag selects the processor. Currently the only options are `pitch` for pitch shifting and `bypass` to bypass the phase vocoder. For example:
```
python pypv.py -p bypass
```
> Bypasses the PV

These flags can mixed, for example:
```
python pypv.py -s sdr 90.4 -p bypass
```
> Tunes the SDR to 90.4MHz and bypasses the PV - functionally a real-time CLI radio client!

