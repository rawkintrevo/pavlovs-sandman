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
