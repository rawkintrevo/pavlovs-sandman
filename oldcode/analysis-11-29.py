import pandas as pd
import matplotlib.pyplot as plt

import os

dateTarget = '2017-11-28'
snorePath = 'training_data/volume_data/snore/%s' % dateTarget
noSnorePath = 'training_data/volume_data/non-snore/%s' % dateTarget

snoreCsvs = os.listdir(snorePath)
nonSnoreCsvs = os.listdir(noSnorePath)

## Quick Inspection
howManyPlots = 4
for i in range(howManyPlots):
    pd.DataFrame \
        .from_csv("%s/%s" % (noSnorePath, nonSnoreCsvs[i]))\
        .plot(title = "Non Snore @ %s " % nonSnoreCsvs[i].replace(".csv",""))

for i in range(howManyPlots):
    pd.DataFrame \
        .from_csv("%s/%s" % (snorePath, snoreCsvs[i]))\
        .plot(title = "Snore @ %s" % snoreCsvs[i].replace(".csv", ""))


## Filter this mess

from scipy.ndimage.filters import gaussian_filter

# I got 10 by inspection

for i in range(howManyPlots):
    noisyDf = pd.DataFrame \
           .from_csv("%s/%s" % (snorePath, snoreCsvs[i+45]))
    noisyDf['gausFilter'] = gaussian_filter(noisyDf['value'],   50)
    noisyDf = noisyDf.reset_index()
    noisyDf['gausFilter'].plot(title = "Smoothed Snores Sample")

for i in range(howManyPlots):
    noisyDf = pd.DataFrame \
        .from_csv("%s/%s" % (noSnorePath, nonSnoreCsvs[i+10]))
    noisyDf['gausFilter'] = gaussian_filter(noisyDf['value'], 50)
    noisyDf = noisyDf.reset_index()
    noisyDf['gausFilter'].plot(title = "Smoothed Non-Snores Sample")

### Do it again, "fit" some Sin waves
## Thanks this guy
# https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy

import numpy as np
from scipy.optimize import leastsq

def fitSinWav(data, tDiv=80):
    N = data.size
    tFactor = N / tDiv # this is somewhat bounded by how often you breath
    t = np.linspace(0, tFactor*np.pi, N)
    guess_mean = np.mean(data)
    guess_std = 7*np.std(data)/(2**0.5)
    guess_phase = 0
    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(t+x[1]) + x[2] - data
    est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]
    # recreate the fitted curve using the optimized parameters
    data_fit = est_std*np.sin(t+est_phase) + est_mean
    return data_fit


for i in range(howManyPlots):
    noisyDf = pd.DataFrame \
        .from_csv("%s/%s" % (snorePath, snoreCsvs[i]))
    noisyDf['gausFilter'] = gaussian_filter(noisyDf['value'], 10)
    noisyDf = noisyDf.reset_index()
    noisyDf['fitSinWave70'] = fitSinWav(noisyDf['gausFilter'], 70)
    noisyDf['fitSinWave80'] = fitSinWav(noisyDf['gausFilter'])
    noisyDf['fitSinWave90'] = fitSinWav(noisyDf['gausFilter'], 90)
    noisyDf['fitSinWave100'] = fitSinWav(noisyDf['gausFilter'], 100)
    noisyDf[['gausFilter',
             'fitSinWave70',
             'fitSinWave80',
             'fitSinWave90',
             'fitSinWave100',]].plot(title = "Smoothed Snores Sample @ %s" % snoreCsvs[i].replace(".csv", ""))


for i in range(howManyPlots):
    noisyDf = pd.DataFrame \
        .from_csv("%s/%s" % (snorePath, snoreCsvs[i]))
    noisyDf['gausFilter'] = gaussian_filter(noisyDf['value'], 20)
    noisyDf = noisyDf.reset_index()
    noisyDf['fitSinWave70'] = fitSinWav(noisyDf['gausFilter'], 70)
    noisyDf['fitSinWave75'] = fitSinWav(noisyDf['gausFilter'], 75)
    noisyDf['fitSinWave80'] = fitSinWav(noisyDf['gausFilter'], 80)
    noisyDf['fitSinWave85'] = fitSinWav(noisyDf['gausFilter'], 85)
    noisyDf[['gausFilter',
             'fitSinWave70',
             'fitSinWave85',
             'fitSinWave80',
             'fitSinWave85',]].plot(title = "Smoothed Snores Sample @ %s" % snoreCsvs[i].replace(".csv", ""))

for i in range(howManyPlots):
    noisyDf = pd.DataFrame \
        .from_csv("%s/%s" % (noSnorePath, nonSnoreCsvs[i]))
    noisyDf['gausFilter'] = gaussian_filter(noisyDf['value'], 10)
    noisyDf = noisyDf.reset_index()
    noisyDf['fitSinWave70'] = fitSinWav(noisyDf['gausFilter'], 70)
    noisyDf['fitSinWave75'] = fitSinWav(noisyDf['gausFilter'], 75)
    noisyDf['fitSinWave80'] = fitSinWav(noisyDf['gausFilter'], 80)
    noisyDf['fitSinWave85'] = fitSinWav(noisyDf['gausFilter'], 85)
    noisyDf[['gausFilter',
             'fitSinWave70',
             'fitSinWave85',
             'fitSinWave80',
             'fitSinWave85',]].plot(title = "Smoothed Non Snores Sample @ %s" % snoreCsvs[i].replace(".csv", ""))

    ##############
    ## Interesting Observations so far
    #   The temporal component-
    #       snores occur at interval, that helps make them identifiable (dictated by breath
    #       problem: someitmes i snore on inhale and excale, sometimes only one or the other
    #
    #   "Non snores identifiable by relatively small distance between peaks and valleys
    #
    #   There's probably some mis labled samples
    #
    #   Maybe something here with Sin Wav fit- but going to take a lot more preprocessing
    #       my gut tells me to accept what I've learned and continue on my journey, maybe I'll come
    #       back some day.
    #
    #   Kudos:
    #       1. I believe this problem can be solved in very small dimension space
    #           (e.g. clustering on maybe 3-5 dimensions)
    #       2. I don't think I'll need to invoke specialty audio analysis libraries (but I will anyway)
    #       3. I won't need to turn to dark magic (CNNs, though I may anyway, time permitting)


from scipy.signal import argrelextrema

for i in range(howManyPlots):
    noisyDf = pd.DataFrame \
        .from_csv("%s/%s" % (snorePath, snoreCsvs[i]))
    print(snoreCsvs[i])
    x = gaussian_filter(noisyDf['value'], 9)
    # for local maxima
    print("local maxima: ", argrelextrema(x, np.greater))
    # for local minima
    print("local minima: ", argrelextrema(x, np.less))

indicesOfInterst = np.ndarray.tolist(np.sort(np.concatenate((argrelextrema(x, np.less), argrelextrema(x, np.greater)), 1)))[0]
maxima = x[indicesOfInterst]
waterLine = x.mean()

diffs = np.ndarray.tolist(np.diff(x[indicesOfInterst]))

thresh = 75
finals = [x[indicesOfInterst[i]] for i in range(len(indicesOfInterst)-1) if abs(diffs[i]) > thresh]
np.array(finals)[np.concatenate((argrelextrema(np.array(finals), np.less), argrelextrema(np.array(finals), np.greater)), 1)]

for i in range(howManyPlots):
    print(snoreCsvs[i])
    noisyDf = pd.DataFrame \
        .from_csv("%s/%s" % (snorePath, snoreCsvs[i]))
    for c in range(10,30):
        noisyDf['gausFilter']  = gaussian_filter(noisyDf['value'], c)
        noisyMaxima = argrelextrema( gaussian_filter(noisyDf['value'], c), np.greater)
        sinWavMaxima = argrelextrema(fitSinWav(noisyDf['gausFilter'], 80), np.greater)
        if len(noisyMaxima[0]) == len(sinWavMaxima[0]):
            print(c)
            print(noisyMaxima[0] - sinWavMaxima[0])
            break
