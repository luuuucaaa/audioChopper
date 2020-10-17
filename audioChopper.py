#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 18:55:06 2020

@author: luca
"""

# =============================================================================
# import libraries
# =============================================================================

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal import butter, lfilter
from numpy import diff
import math
from scipy.signal import find_peaks, peak_prominences, peak_widths

# =============================================================================
# variabels
# =============================================================================

# general
zoomFactor = 1 # determines x-axis limit from 0 (1 for full x-axis)
writeAudioSegments = True
fadeLength = 100 # fade-in and fade-out in samples

# weightning of features for combinedFeatures (s)
drmsWeight = 1 # weight of amplitude derivation (rms (root-mean-square) derivation) 
dsfmWeight = 1 # weight of tonalness derivation (sfm (spectral-flatness-measure) derivation)

# variables for peak detection in combinedFeatures
sensitivity = 2 # minimum distance between cutpoints in seconds
peakHeight = 0.2 # minimum peak height
deltaThreshold = 0.005 # for finding valleys in front of peaks. good value = 0.005. script iterates from peak values backwards; if (value - previous value < deltaThreshold): break.

# audio filter
filterAudio = False
filterOrderAudio = 2
filterCutoffAudio = 300

# rms (root-mean-square) filter
filterRms = False
filterOrderRms = 2
filterCutoffRms = 300

# rms (root-mean-square) derivation filter
filterdRms = True
filterOrderdRms = 1
filterCutoffdRms = 1000

# sfm (spectral flatness measure) filter
filterSfm = True
filterOrderSfm = 2
filterCutoffSfm = 800

# sfm (spectral flatness measure) derivation filter
filterdSfm = True
filterOrderdSfm = 1
filterCutoffdSfm = 1000

# combined features filter
filterCombinedFeatures = True
filterOrderCombinedFeatures = 2
filterCutoffCombinedFeatures = 1800 

# =============================================================================
# constants
# =============================================================================

figureWidth = 24
figureHeight = 20
fs = 44800
audio = []
audioSlices = []

# =============================================================================
# load audio
# =============================================================================

audio.append(librosa.load('testAudio/Musik.wav', sr = fs))
audio.append(librosa.load('testAudio/test.wav', sr = fs))
audio.append(librosa.load('testAudio/test2.wav', sr = fs))
audio.append(librosa.load('testAudio/test3.wav', sr = fs))
audio.append(librosa.load('testAudio/test4.wav', sr = fs))

audio = np.asarray(audio, dtype=object)
audio = audio[:, 0]

# =============================================================================
# filtering / interpolation function definition
# =============================================================================
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# =============================================================================
# audio processing
# =============================================================================

for i, a in enumerate(audio):
    
    # =========================================================================
    # analyze audio
    # =========================================================================
    
    if (filterAudio):
        a = butter_lowpass_filter(a, filterCutoffAudio, fs, filterOrderAudio)
        
    # normalize audio
    a /= np.amax(np.abs(a))
    
    # calculate rms
    hopLength = 512
    frameLength = 1024
    rms = librosa.feature.rms(a, frame_length=frameLength, hop_length=hopLength, center=True)
    rms = rms[0]
    rms /= np.amax(np.abs(rms))
    if (filterRms):
        rms = butter_lowpass_filter(rms, filterCutoffRms, fs, filterOrderRms)
        rms /= np.amax(np.abs(rms))
    
    # calculate rms derivation
    dxrms = 1
    drms = diff(rms) / dxrms
    drms /= np.amax(np.abs(drms))
    if (filterdRms):
        drms = butter_lowpass_filter(drms, filterCutoffdRms, fs, filterOrderdRms)
        drms /= np.amax(np.abs(drms))
    
    # calculate spectral flatness
    s, phase = librosa.magphase(librosa.stft(a))
    sfm = librosa.feature.spectral_flatness(S=s, hop_length=256)
    sfm = sfm[0]
    np.interp(sfm, (sfm.min(), sfm.max()), (0, 1))
    sfm /= np.amax(np.abs(sfm))
    if (filterSfm):
        sfm = butter_lowpass_filter(sfm, filterCutoffSfm, fs, filterOrderSfm)
        sfm /= np.amax(np.abs(sfm))
    
    # calculate spectral flatness derivation
    dxsfm = 1
    dsfm = diff(sfm / dxsfm, axis=0)
    dsfm /= np.amax(np.abs(dsfm))
    if (filterdSfm):
        dsfm = butter_lowpass_filter(dsfm, filterCutoffdSfm, fs, filterOrderdSfm)
        dsfm /= np.amax(np.abs(dsfm))
    
    # =========================================================================
    # combine features
    # =========================================================================
        
    combinedFeatures = (drmsWeight * drms) + (dsfmWeight * np.abs(dsfm))
    np.interp(combinedFeatures, (combinedFeatures.min(), combinedFeatures.max()), (0, 1))
    combinedFeatures /= np.amax(np.abs(combinedFeatures))
    for i in range(0, len(combinedFeatures)):
        if (combinedFeatures[i] < 0):
            combinedFeatures[i] = 0
        else:
            combinedFeatures[i] = combinedFeatures[i]
        
    if (filterCombinedFeatures):
        combinedFeatures = butter_lowpass_filter(combinedFeatures, filterCutoffCombinedFeatures, fs, filterOrderCombinedFeatures)
        combinedFeatures /= np.amax(np.abs(combinedFeatures))
    
    # =========================================================================
    # find cutpoints
    # =========================================================================
    
    peakDistance = sensitivity * fs / hopLength
    peaks, _ = find_peaks(combinedFeatures, height=peakHeight, distance=peakDistance)
    
    prePeakValleys = []
    for p in peaks:
        nextValue = True
        i = 0
        while (nextValue):
            i -= 1
            index = p + i
            A = (combinedFeatures[index]/3 + combinedFeatures[index-1]/3 + combinedFeatures[index-2]/3)
            B = (combinedFeatures[index-1]/3 + combinedFeatures[index-2]/3 + combinedFeatures[index-3]/3)
            delta = A - B
            if (delta < deltaThreshold):
                nextValue = False
            if (index == 0):
                nextValue = False
        prePeakValleys.append(index)
            
    cutpointIndices = np.asarray(prePeakValleys) * hopLength
    cutpointSeconds = cutpointIndices / fs
    
    # =========================================================================
    # plot relevant features
    # =========================================================================
    
    plt.figure(figsize = (figureWidth, figureHeight))
    plt.subplots_adjust(hspace = 0.4)
    time = np.linspace(0, len(a) / fs, num=len(a))
    
    # waveform
    plt.subplot(611)
    plt.title('Waveform')
    plt.plot(time, a)
    plt.vlines(cutpointSeconds, -1.05, 1.05, color='red', linestyle='solid')
    plt.xlim(0, time[-1] * zoomFactor)
    plt.ylim(-1.05, 1.05)
    
    # rms
    plt.subplot(612)
    plt.title('RMS')
    plt.plot(rms)
    plt.xlim(0, len(rms) * zoomFactor)
    plt.ylim(0, 1.05)
    
    # rms derivation
    plt.subplot(613)
    plt.title('RMS Derivation')
    plt.plot(drms)
    plt.xlim(0, len(drms) * zoomFactor)
    plt.ylim(-1.05, 1.05)
    
    # spectral flatness
    plt.subplot(614)
    plt.title('SFM')
    plt.plot(sfm)
    plt.xlim(0, len(sfm) * zoomFactor)
    plt.ylim(0, 1.05)
    
    # spectral flatness derivation
    plt.subplot(615)
    plt.title('SFM Derivation')
    plt.plot(dsfm)
    plt.xlim(0, len(dsfm) * zoomFactor)
    plt.ylim(-1.05, 1.05)
    
    # combined features
    plt.subplot(616)
    plt.title('Combined Features')
    plt.plot(combinedFeatures)
    plt.plot(prePeakValleys, combinedFeatures[prePeakValleys], "rx")
    plt.xlim(0, len(combinedFeatures) * zoomFactor)
    plt.ylim(0, 1.05)
    
    plt.show()
    
    # waveform with cutpoints
    plt.figure(figsize = (figureWidth, figureHeight / 2))
    
    plt.title('Waveform')
    plt.plot(time, a)
    plt.vlines(cutpointSeconds, -1.05, 1.05, color='red', linestyle='solid')
    plt.xlim(0, time[-1] * zoomFactor)
    plt.ylim(-1.05, 1.05)
    
    plt.show()
    
    # =========================================================================
    # slice audio into segments
    # =========================================================================

    startPos = 0
    for c in cutpointIndices:
        x = a[startPos:c]
        x /= np.amax(np.abs(x))
        x = np.pad(x, (1, 1), 'constant')
        audioSlices.append(x)
        startPos = c
    # for last segment
    x = a[startPos:]
    x /= np.amax(np.abs(x))
    x = np.pad(x, (1, 1), 'constant')
    audioSlices.append(x)
    
    # add fades
    for s in audioSlices:
        for i in range(0, fadeLength):
            s[i] *= i / fadeLength
            s[-i] *= i / fadeLength
    
    # =========================================================================
    # write audio
    # =========================================================================
    
    if (writeAudioSegments):
        for s in audioSlices:
            segmentID = str(s[1]) + str(s[2]) + str(s[4]) + '0000000000000000000000000000000'
            filename = segmentID.replace(' ', '')
            filename = filename.replace('.', '')
            filename = filename.replace('[', '')
            filename = filename.replace(']', '')
            filename = filename.replace('-', '')
            filename = filename.replace('e', '')
            filename = filename[:30]
            filename = './cutpointFinder_generatedAudio/' + filename + '.wav'
            librosa.output.write_wav(filename, s, fs)