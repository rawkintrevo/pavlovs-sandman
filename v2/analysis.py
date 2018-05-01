
import numpy as np
import os
import matplotlib.pyplot as plt

from pyAudioAnalysis.audioFeatureExtraction import stFeatureExtraction
from pyAudioAnalysis import audioBasicIO

path = "/home/rawkintrevo/gits/pavlovs-sandman/training_data/recordings/snore/"

fileName = os.listdir(path)[0]
print(fileName)
[Fs, data] = audioBasicIO.readAudioFile("%s%s" % (path, fileName ))
# for wav in os.listdir(path)[1:3]:
#     [Fs, x] = audioBasicIO.readAudioFile("%s%s" % (path, wav))
#     data = np.concatenate((data, x), 0)



# 0 : 1500, 2000
# 1:  8000, 15000
# 2:  a bit with 1500, 2000 (was light snoring)
# 3: 1500, 2000
# 4: 8000, 15000
# 5: 1500, 2000
# 6: 5000, 7000 # a little but not much on 8k-15k
# 7: ?
# 8: 5000, 7000 (but 8-15 also is OK)
# 9 : 1500, 2000
# 10: 5000, 7000
# 11: 5k, 7k
# 12: 1500, 2k
# 13: 1500, 2k, very light snoring
# 14: 6k, 10k (dog kicking around in background, need this range to really dial it in
# 15: ? light snoring
# 16: 7k, 12k
# 17: 1500, 2k
# 18: 1500, 2k

def filterFreqs(data, lowpass=0, highpass=22050, filters= [], stereo= False):
    """
    Based entirely on https://rsmith.home.xs4all.nl/miscellaneous/filtering-a-sound-recording.html
    :param freqs: Filter everything above highpass, below lowpass, and between each tuple element in filters
    :param stereo: Set to true if listening to two channel
    :return: (data with filtered out freqs, intensities of various freqs)
    """
    if stereo:
        x = data[0::2]
    else:
        x = data
    fft = np.fft.rfft(x)
    fft[:lowpass] = 0
    fft[highpass:] = 0
    for filter in filters:
        fft[filter[0]: filter[1]] = 0
    ns = np.fft.irfft(fft).astype(np.int16)
    return (ns, fft)
    #np.fft.rfft real-fast Fourier Transform


def plotStuff(data_orig, data_filtered, fft_orig, fft):
    data_mean = np.mean(np.abs(data_filtered))
    data_std = np.std(np.abs(data_filtered))
    plt.figure(1)
    a = plt.subplot(211)
    r = (2**16/2)/8
    a.set_ylim([-r, r])
    a.set_xlabel('time [s]')
    a.set_ylabel('sample value [-]')
    a.set_ylim(0)
    x = np.linspace(0, len(data_orig)/44100, len(data_orig))
    plt.plot(x, data_orig),
    plt.plot(x, data_filtered)
    # plt.plot(x, ma)
    # a.axhline(data_mean+ 1.5*data_std, color='k')
    b = plt.subplot(212)
    b.set_xscale('log')
    b.set_xlabel('frequency [Hz]')
    b.set_ylabel('|amplitude|')

    b.set_ylim(0, fft.max())
    plt.plot(abs(fft_orig))
    plt.plot(abs(fft))
    print(fileName)


fileName = os.listdir(path)[0]
print(fileName)
[Fs, data] = audioBasicIO.readAudioFile("%s%s" % (path, fileName ))

data_orig, fft_orig = filterFreqs(data)
data_filtered, fft = filterFreqs(data, lowpass= 1500,
    highpass= 2050)


data_filtered, fft = filterFreqs(data, lowpass= 1500,
                                 highpass= 15000,
                                 filters=[(2001,7999)])

plotStuff(data_orig, data_filtered, fft_orig, fft)

## data_filtered is a wav- need to just get peaks and not where they are > whatev
plt.plot(data_filtered[0:4000])
plt.plot(np.abs(data_filtered[0:4000]))




data_mean = np.mean(np.abs(data_filtered))
data_std = np.std(np.abs(data_filtered))

## Create "snore" series, and the non-snore series inbetween. Measure they're mean/std dev
## Define a snore as something that is within specs

def durations(data_filtered, verbose=False):
# look at max over 10/th second blocks
    data_mean = np.mean(np.abs(data_filtered))
    data_std = np.std(np.abs(data_filtered))
    thresh = data_mean + 1.5*data_std
    snore = False
    activeSeq = []
    output = []
    for i in range(0, len(data_filtered), 4410):
        val = np.max(data_filtered[i: i+4410])
        if val > thresh:
            if snore == False:
                if verbose:
                    print("noSnore len: ", len(activeSeq)/10)
                output.append(("ns", len(activeSeq)/10))
                snore = True
                activeSeq = [val]
            else:
                activeSeq.append(val)
        if val < thresh:
            if snore == True:
                if verbose:
                    print("snore len: ", len(activeSeq)/10)
                    print(activeSeq)
                output.append(("s", len(activeSeq)/10))
                snore = False
                activeSeq = [0]
            else:
                activeSeq.append(0)
    return output

## Any snore of 0.1 should be turned into a "noSnore" and then merge the adjacent "noSnores"
def parseDurrationsForSnores(durrs):
    snore_min = 0.4
    snore_max = 1.3
    nsnore_min = 2.1
    nsnore_max = 4.0
    count = 0
    for i in range(0, len(durrs) -2):
        if durrs[i][0] == 's' and durrs[i][1] < snore_max and durrs[i][1] > snore_min:
            if durrs[i+1][1] > 2.2 and durrs[i+1][1] < 4.0:
                if durrs[i + 2][1] > snore_min and durrs[i + 2][1] < snore_max:
                    print(i, "snore detected", durrs[i], durrs[i+1], durrs[i+2])
                    count += 1
    return count


for fileName in os.listdir(path):
    # for i in [0,3,5]:
    #     fileName = os.listdir(path)[i]
    [Fs, data] = audioBasicIO.readAudioFile("%s%s" % (path, fileName ))
    data_orig, fft_orig = filterFreqs(data)
    data_filtered1, fft = filterFreqs(data, lowpass= 1500, highpass=2000)
    data_filtered2, fft = filterFreqs(data, lowpass= 8000, highpass=15000)
    # plotStuff(data_orig, data_filtered, fft_orig, fft)
    print("----- %s --- Low Band" % fileName)
    durrs1 = durations(data_filtered1)
    c1 = parseDurrationsForSnores(durrs1)
    print("----- %s --- High Band" % fileName)
    durrs2 = durations(data_filtered2)
    c2 = parseDurrationsForSnores(durrs2)
    if c1 == 0 and c2== 0:
        pass
fileName = "1-59-54.wav"

durrs1 = durations(data_filtered1, True)
plotStuff(data_orig, data_filtered1, fft, fft)

def snoreOrNo(data):
    data_filtered1, fft = filterFreqs(data, lowpass= 1500, highpass=2000)
    data_filtered2, fft = filterFreqs(data, lowpass= 8000, highpass=15000)
    # plotStuff(data_orig, data_filtered, fft_orig, fft)
    print("----- %s --- Low Band" % fileName)
    durrs1 = durations(data_filtered1)
    c1 = parseDurrationsForSnores(durrs1)
    print("----- %s --- High Band" % fileName)
    durrs2 = durations(data_filtered2)
    c2 = parseDurrationsForSnores(durrs2)
    return (c1, c2)