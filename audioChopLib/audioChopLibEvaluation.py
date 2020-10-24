#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# import libraries
# =============================================================================

from audioChopLib import *

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# create test audio
# =============================================================================

# set sampling frequency
fs = 48000

# # the frequencies of the sine signals are chosen to fir the middle frequencies
# # of the bandpass filters of the auditory filterbank

# scaling factor
c = 0.1618

# bandwith of lowest bandpass
dfb = 81.9289

# critical bandrate scales
z = []
zz = 0
while (zz < 26.5):
    zz += 0.5
    z.append(zz)

# calculate middle frequency fm for each bandpass filter
fsin = np.array(z)
for i in range(len(z)):
    fsin[i] = (dfb / c) * np.sinh(c * z[i])

# define length of sine signals
lsin = 2 * fs

# create sample number array for sine signals
nsin = np.arange(0, lsin, 1)

# calculate frequency array from sample number array
f = np.fft.fftfreq(nsin.shape[-1])

# scale frequency array from values between 0 and 1  to values between 0 and fs
f *= fs

# initialize array for sine time domain signals
sin = np.zeros(lsin)

# initialize array for sine frequency domain signals
sinfft = np.zeros(lsin, dtype=complex)

# calculate sine signals
for i in range(len(fsin)):

    # calculate time domain signal
    sin_ = 1 * np.sin(2 * np.pi * fsin[i] * (nsin / fs))

    # normalize time domain signal
    sin_ /= np.amax(np.abs(sin_))

    # add time domain signals together
    sin += sin_

    # calculate frequency domain signal from time domain signal
    sinfft_ = np.fft.fft(sin_)

    # normalize frequency domain signal
    sinfft_ /= np.amax(np.abs(sinfft_))

    # add frequency domain signals together
    sinfft += sinfft_

# normalize time domain signal array
sin /= np.amax(np.abs(sin))

# normalize frequency domain signal array
sinfft /= np.amax(np.abs(sinfft))

# calculate dB-values from absolute values of frequency domain signal
h = 20 * np.log10(abs(sinfft))

# create plot
figWidth = 20
figHeight = 14
plt.figure(figsize = (figWidth, figHeight))
plt.subplots_adjust(hspace=0.2)

# plot time domain signal
plt.subplot(2, 1, 1)
plt.plot(nsin, sin, linewidth=0.5)
plt.title('Time Signal of of Evaluation-Signal (unfiltered)')
plt.xlabel('Samples')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', linestyle='-', color='#cccccc')
plt.xlim([0, len(nsin)])
plt.ylim([-1.05, 1.05])

# plot frequency domain signal
plt.subplot(2, 1, 2)
plt.semilogx(f, h, linewidth=1)
plt.title('Frequency Response of Evaluation-Signal (unfiltered)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', linestyle='-', color='#cccccc')
plt.xlim([20, 20000])
plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
plt.ylim([-30, 5])

plt.show()

# =============================================================================
# evaluate library functions
# =============================================================================

# filter evaluation signal with ear filter
sinEarFilter = ear_filter(sin, plot=True)

# filter evaluation signal with auditory filter
sinAuditoryFilter, sinPerceivedLoudness = auditory_filter(sinEarFilter, plot=True)