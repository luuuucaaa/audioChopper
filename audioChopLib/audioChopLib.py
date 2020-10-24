#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# import libraries
# =============================================================================

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# =============================================================================
# define functions
# =============================================================================

def init_ear_filter(fs=48000):

    # b-coefficients of ear filter
    b = [[+1.0159, -1.9253, +0.9221], [+0.9589, -1.8061, +0.8764],
         [+0.9614, -1.7636, +0.8218], [+2.2258, -1.4347, -0.4982],
         [+0.4717, -0.3661, +0.2441], [+0.1153, +0.0000, -0.1153],
         [+0.9880, -1.9124, +0.9261], [+1.9522, +0.1623, -0.6680]]

    # a-coefficients of ear filter
    a = [[+1.0000, -1.9253, +0.9380], [+1.0000, -1.8061, +0.8354],
         [+1.0000, -1.7636, +0.7832], [+1.0000, -1.4347, +0.7276],
         [+1.0000, -0.3661, -0.2841], [+1.0000, -1.7960, +0.8058],
         [+1.0000, -1.9124, +0.9142], [+1.0000, +0.1623, +0.2842]]

    # combine all b-coefficients to B and all a-coefficients to A
    B = b[0]
    A = a[0]
    for i in range(7):
        B = np.convolve(B, b[i + 1])
        A = np.convolve(A, a[i + 1])
        
    # create frequency-array
    f = np.arange(20, 20000, 1)

    # calculate angular frequency from frequency-array
    w = 2 * np.pi * f / fs

    # calculate frequency-response of ear filter with A, B and w
    w, h = signal.freqz(B, A, w)
    
    # normalize absolute values h
    h /= np.amax(np.abs(h))

    # calculate dB-values H from absolute values h
    H = 20 * np.log10(abs(h))

    # output
    # B: 1D array of filter b-coefficients
    # A: 1D array of filter a-coefficients
    # f: 1D array of frequency-values
    # H: 1D array of dB-values
    return B, A, f, H

def ear_filter(data, fs=48000, plot=False, figWidth=20, figHeight=14):
    # data: 1D array of 1 channel time domain data series

    # initialize ear filter
    B, A, f, H = init_ear_filter(fs)

    # filtering data with initialized ear filter
    y = signal.lfilter(B, A, data)
    
    # normalize time domain signal array
    y /= np.amax(np.abs(y))
    
    # create sample array for data
    ny = np.arange(0, len(y), 1)
    
    # calculate frequency domain signal from time domain signal
    yfft = np.fft.fft(y)
    
    # normalize frequency domain signal array
    yfft /= np.amax(np.abs(yfft))
    
    # calculate dB-values from absolute values of frequency domain signal
    Hyfft = 20 * np.log10(abs(yfft))
    
    # calculate frequency array from sample array
    nyfft = np.fft.fftfreq(ny.shape[-1])
    
    # scale frequency array from values between 0 and 1  to values between 0 and fs
    nyfft *= fs
    
    # plot
    if (plot):
        
        # create plot
        plt.figure(figsize = (figWidth, figHeight))
        plt.subplots_adjust(hspace=0.2)
        
        # plot time domain signal
        plt.subplot(2, 1, 1)
        plt.plot(ny, y, linewidth=0.5)
        plt.title('Time Signal of Evaluation-Signal (ear filtered)')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude [dB]')
        plt.grid(which='both', linestyle='-', color='#cccccc')
        plt.xlim([0, len(ny)])
        plt.ylim([-1.05, 1.05])
        
        # plot time frequency signal
        plt.subplot(2, 1, 2)
        plt.title('Frequency Response of Evaluation-Signal (ear filtered)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.semilogx(nyfft, Hyfft, label='Data Frequency Response')
        plt.semilogx(f, H, label='Target Frequency Response')
        plt.legend(loc="upper left")
        plt.grid(which='both', linestyle='-', color='#cccccc')
        plt.xlim([20, 20000])
        plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        plt.ylim([-30, 5])
        plt.show()

    # output
    # y: 1D array of 1 channel time domain data series
    return y

def rectification(datafft):
    # data: 1D array of 1 channel frequency domain data series
    
    # calculate time signal of impulse response
    data = np.fft.ifft(datafft)

    # rectification loop
    for i in range(len(data)):
        if (data[i] > 0):
            data[i] = data[i]
        elif (data[i] <= 0):
            data[i] = 0

    # calculate impulse response from time signal
    y = np.fft.fft(data)

    # factoring frequency response for compensating energy loss through rectification
    y *= 2
    
    # output
    # y: 1D array of 1 channel frequency domain data series
    return y

def segmentation(data, calculateRms=False):
    # data: 2D array of 53 channel time domain data series
    
    # define block and hop lengths for segmentation
    blockLength = [8192, 4096, 2048, 1024]
    hopLength = [2048, 1024, 512, 256]
    
    # initialize array for segments which will be returned for further calculations
    segmentBank = []

    if (calculateRms):
        # initialize array for band-specific root-mean-square (RMS) values
        rmsBank = []

    # calculation loop for band-specific root-mean-square (RMS) values
    for i in range(len(data)):

        # get length of time signal
        length = len(data[i])

        # band-specific segmentation of time signals
        if (i < 3):

            # initialize index
            index = 0

            # initialize sub-array for segments
            segmentBank2048 = []

            if (calculateRms):
                # initialize sub-array for band-specific root-mean-square (RMS) values
                rmsBank2048 = []

            # segmentation loop
            while (index * hopLength[0] < length):

                # set start of segment
                start = index * hopLength[0]

                # set end of segment
                end = index * hopLength[0] + blockLength[0]
                if (end > length - 1):
                    end = length - 1

                # get segment
                segment = data[i][start:end]
                
                # append segment to segment bank
                segmentBank2048.append(segment)

                if (calculateRms):
                    # calculate rms value of segment
                    rms = np.sqrt(np.mean(np.abs(segment) ** 2, axis=0, keepdims=True))
    
                    # append rms value to rms sub-array
                    rmsBank2048.append(rms)

                # raise index
                index += 1
                
            # append segment sub-array to segment array
            segmentBank.append(segmentBank2048)

            if (calculateRms):
                # append rms sub-array to rms array
                rmsBank.append(rmsBank2048)

        elif (i >= 3 and i < 16):

            # initialize index
            index = 0

            # initialize sub-array for segments
            segmentBank1024 = []

            if (calculateRms):
                # initialize sub-array for band-specific root-mean-square (RMS) values
                rmsBank1024 = []

            # segmentation loop
            while (index * hopLength[1] < length):

                # set start of segment
                start = index * hopLength[1]

                # set end of segment
                end = index * hopLength[1] + blockLength[1]
                if (end > length - 1):
                    end = length - 1

                # get segment
                segment = data[i][start:end]
                
                # append segment to segment bank
                segmentBank1024.append(segment)

                if (calculateRms):
                    # calculate rms value of segment
                    rms = np.sqrt(np.mean(np.abs(segment) ** 2, axis=0, keepdims=True))
    
                    # append rms value to rms bank
                    rmsBank1024.append(rms)

                # raise index
                index += 1
                
            # append segment sub-array to segment array
            segmentBank.append(segmentBank1024)

            if (calculateRms):
                # append rms sub-array to rms array
                rmsBank.append(rmsBank1024)

        elif (i >= 16 and i < 25):

            # initialize index
            index = 0

            # initialize sub-array for segments
            segmentBank512 = []

            if (calculateRms):
                # initialize sub-array for band-specific root-mean-square (RMS) values
                rmsBank512 = []

            # segmentation loop
            while (index * hopLength[2] < length):

                # set start of segment
                start = index * hopLength[2]

                # set end of segment
                end = index * hopLength[2] + blockLength[2]
                if (end > length - 1):
                    end = length - 1

                # get segment
                segment = data[i][start:end]
                
                # append segment to segment bank
                segmentBank512.append(segment)

                if (calculateRms):
                    # calculate rms value of segment
                    rms = np.sqrt(np.mean(np.abs(segment) ** 2, axis=0, keepdims=True))
    
                    # append rms value to rms bank
                    rmsBank512.append(rms)

                # raise index
                index += 1
                
            # append segment sub-array to segment array
            segmentBank.append(segmentBank512)

            if (calculateRms):
                # append rms sub-array to rms array
                rmsBank.append(rmsBank512)

        else:

            # initialize index
            index = 0

            # initialize sub-array for segments
            segmentBank256 = []

            if (calculateRms):
                # initialize sub-array for band-specific root-mean-square (RMS) values
                rmsBank256 = []

            # segmentation loop
            while (index * hopLength[3] < length):

                # set start of segment
                start = index * hopLength[3]

                # set end of segment
                end = index * hopLength[3] + blockLength[3]
                if (end > length - 1):
                    end = length - 1

                # get segment
                segment = data[i][start:end]
                
                # append segment to segment bank
                segmentBank256.append(segment)

                if (calculateRms):
                    # calculate rms value of segment
                    rms = np.sqrt(np.mean(np.abs(segment) ** 2, axis=0, keepdims=True))
    
                    # append rms value to rms bank
                    rmsBank256.append(rms)

                # raise index
                index += 1
                
            # append segment sub-array to segment array
            segmentBank.append(segmentBank2048)

            if (calculateRms):
                # append rms sub-array to rms array
                rmsBank.append(rmsBank256)
        
    # transform list to array for further calculations
    segmentBank = np.asarray(segmentBank, dtype=object)

    if (calculateRms):
        # transform list to array for further calculations
        rmsBank = np.asarray(rmsBank, dtype=object)
    
    # output
    # segmentBank: 3D array of 53 channel time domain data series, segmented into blocks
    # rmsBank: 2D array of 53 channel rms values
    if (calculateRms):
        return segmentBank, rmsBank
    else:
        return segmentBank

def init_auditory_filterbank(order=5, fs=48000, rectificate=True):

    # calculate nyquist-frequency
    nq = fs / 2

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
    fm = np.array(z)
    for i in range(len(z)):
        fm[i] = (dfb / c) * np.sinh(c * z[i])

    # calculate bandwidth fb for each bandpass filter
    fb = np.array(z)
    for i in range(len(z)):
        fb[i] = np.sqrt((dfb) ** 2 + (c * fm[i]) ** 2)

    # frequency-array
    f = np.arange(20, 20000, 1)
    
    # initilaize dB-value array
    H = []

    # initialize array for low-cut frequencies
    fLow = np.zeros(len(z))
    
    # initialize array for high-cut frequencies
    fHigh = np.zeros(len(z))
    
    # initialize bandpass filterbank
    sosFilterbank = []

    # calculation loop for bandpass filterbank
    for i in range(len(z)):

        # low-cuts of bandpass filters
        fLow[i] = (fm[i] - fb[i]/2) / nq
        if (fLow[i] <= 0): fLow[i] = 0.00001

        # high-cuts of bandpass filters
        fHigh[i] = (fm[i] + fb[i]/2) / nq
        if (fHigh[i] <= 0): fHigh[i] = 0.00001

        # create bandpass filter with given low- and high-cuts
        sos = signal.butter(order, [fLow[i], fHigh[i]], btype='band', output='sos')

        # calculate angular frequency from frequency-array
        w = 2 * np.pi * f / fs

        # calculate frequency-response of auditory filter with sos and w
        w, h = signal.sosfreqz(sos, w)

        # rectify bandpass signal
        if (rectificate):
            rectification(h)
            
        # calculate dB-values H from absolute values h
        H_ = 20 * np.log10(abs(h))
        
        # append dB-values to dB-value array
        H.append(H_)

        # append bandpass to bandpass filter bank
        sosFilterbank.append(sos)

    # output
    # sosFilterbank: 1D array of filter sos-coefficients for each bandpass filter
    # fm: 1D array of middle frequencies for each bandpass filter
    # f: 1D array of frequencies for plotting
    # H: 2D array of dB-values for each bandpass filter 
    return sosFilterbank, fm, f, H

def auditory_filter(data, fs=48000, plot=False, figWidth=20, figHeight=14):
    # data: 1D array of 1 channel time domain data series

    # initialize ear filter
    sosFilterbank, fm, f, H = init_auditory_filterbank()

    # initialize output array for filtered time signals
    y = []
    
    # initialize array for filtered frequency domain signals
    yfft = []
    
    # filter and calculation loop
    for i in range(len(sosFilterbank)):
        
        # filter time signal with initialized ear filter
        y_ = signal.sosfilt(sosFilterbank[i], data)
        
        # append filtered time signal to filtered time signal array
        y.append(y_)
        
        # calculate frequency domain signal from time signal
        yfft_ = np.fft.fft(y[i])
        
        # sppend frequency domain signal to frequency domain signal array
        yfft.append(yfft_.real)
        
    # normalize output array for filtered time signals
    y /= np.amax(np.abs(y))
        
    # normalize array for filtered frequency domain signals
    yfft /= np.amax(np.abs(yfft))
        
    # create sample number array for time signals
    ny = np.arange(0, len(y[0]), 1)
    
    # calculate frequency array from sample array for frequency domain signal
    nyfft = np.fft.fftfreq(ny.shape[-1])
    
    # scale frequency array from values between 0 and 1  to values between 0 and fs
    nyfft *= fs
    
    # segmentation and rms calculation
    segmentBank, rmsBank = segmentation(y, calculateRms=True)
        
    # sound pressure reference value
    p0 = 0.00002

    # sound pressure level threshold values
    Lpt = [15, 25, 35, 45, 55, 65, 75, 85]

    # calculate sound pressure threshold values from sound pressure level threshold values
    pt = np.zeros(len(Lpt))
    for i in range(len(Lpt)):
        pt[i] = p0 * 10 ** (Lpt[i] / 20)

    # scaling factor
    a = 1.5

    # normalization constant
    c = 0.0217406

    # exponents
    v = [0.0000, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434]

    # initialize array for band-specific perceived loudness values
    perceivedLoudnessBank = []

    # initilaize constant for following calculation
    product = 1

    # calculation loop for band-specific perceived loudness values from rms values
    for i in range(len(rmsBank)):

        # initialize sub-array for band-specific perceived loudness values
        perceivedLoudnessSubBank = []

        for j in range(len(rmsBank[i])):
            for k in range(8):

                # calculate sub-result
                product *= (1 + (rmsBank[i][j] / pt[k]) ** a) ** ((v[k + 1] - v[k]) / a)

            # calculate result
            result = c * (rmsBank[i][j] / p0) * product

            # append result to sub-array for band-specific perceived loudness values
            perceivedLoudnessSubBank.append(result)

            # reset sub-result
            product = 1

        # append sub-array to array for band-specific perceived loudness values
        perceivedLoudnessBank.append(perceivedLoudnessSubBank)

    # lower hearing threshold for each bandpass filter
    lht = [0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453, 0.0365, 0.0298,
           0.0247, 0.0207, 0.0176, 0.0151, 0.0131, 0.0115, 0.0103, 0.0093,
           0.0086, 0.0081, 0.0077, 0.0074, 0.0073, 0.0072, 0.0071, 0.0072,
           0.0073, 0.0074, 0.0076, 0.0079, 0.0082, 0.0086, 0.0092, 0.0100,
           0.0109, 0.0122, 0.0138, 0.0157, 0.0172, 0.0180, 0.0180, 0.0177,
           0.0176, 0.0177, 0.0182, 0.0190, 0.0202, 0.0217, 0.0237, 0.0263,
           0.0296, 0.0339, 0.0398, 0.0485, 0.0622]

    # modify specific loudness with lower hearing threshold
    for i in range(len(perceivedLoudnessBank)):
        for j in range(len(perceivedLoudnessBank[i])):
            if (perceivedLoudnessBank[i][j] >= lht[i]):
                perceivedLoudnessBank[i][j] -= lht[i]
            else:
                perceivedLoudnessBank[i][j] = 0
                
    # get mean perceived loudness for each bandpassed signal
    perceivedLoudnessMean = []
    for i in range(len(perceivedLoudnessBank)):
        perceivedLoudnessMean_ = sum(perceivedLoudnessBank[i]) / len(perceivedLoudnessBank[i])
        perceivedLoudnessMean.append(perceivedLoudnessMean_)
    
    # normalize mean perceived loudness array
    perceivedLoudnessMean /= np.amax(np.abs(perceivedLoudnessMean))
    
    # flip array to get equal-loudness contour from mean perceived loudness
    equalLoudnessContour = np.zeros(len(perceivedLoudnessMean))
    for i in range(len(perceivedLoudnessMean)):
        equalLoudnessContour[i] = 1 - perceivedLoudnessMean[i]
                
    # plot
    if (plot):
        
        # create plot
        plt.figure(figsize = (figWidth, figHeight))
        plt.subplots_adjust(hspace=0.2)
        
        # plot frequency domain signal
        plt.subplot(2, 1, 1)
        for i in range(len(yfft)):
            h = 20 * np.log10(abs(yfft[i]))
            plt.semilogx(nyfft, h, linewidth=0.5)
        plt.title('Frequency Response of all 53 Bands of Evaluation-Signal (auditory filtered)')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude [dB]')
        plt.grid(which='both', linestyle='-', color='#cccccc')
        plt.xlim([20, 20000])
        plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        plt.ylim([-30, 5])
        
        # plot equal-loudness contour
        plt.subplot(2, 1, 2)
        plt.title('Perceived Loudness of Evaluation-Signal (auditory filtered)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.semilogx(fm, perceivedLoudnessMean, linewidth=1, label='Mean Perceived Loudness of Evaluation-Signal ')
        plt.semilogx(fm, equalLoudnessContour, linewidth=1, linestyle='--', label='Equal Loudness Contour of Evaluation-Signal ')
        plt.legend(loc="center left")
        plt.grid(which='both', linestyle='-', color='#cccccc')
        plt.xlim([20, 20000])
        plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        plt.show()

    # output
    # y: 2D array of 53 channel time domain data series
    # perceivedLoudnessBank: 2D array of 53 channel perceived loudness values
    return y, perceivedLoudnessBank
    