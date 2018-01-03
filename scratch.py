import alsaaudio, wave, numpy as np, pandas as pd


import pyAudioAnalysis.audioFeatureExtraction as aF

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)

f = wave.open('training_data/recordings/mixed/3-3-7.wav') #'training_data/recordings/snore/1-6-34.wav')

def snr(singleChannel):
    norm = singleChannel / (max(np.amax(singleChannel), -1 * np.amin(singleChannel)))
    m = norm.mean()
    sd = norm.std()

    return m / sd

def wienerFilter(sample):
    """

    :param sample: array_like
    :return:
    """
    return snr(sample) / (1+ snr(sample))

def play(device, f):
    features1 = ["Zero Crossing Rate",
                 "Energy",
                 "Entropy of Energy",
                 "Spectral Centroid",
                 "Spectral Spread",
                 "Spectral Entropy",
                 "Spectral Flux",
                 "Spectral Rolloff"]
    features2 =  ["MFCC_%i" % i for i in range(0, 13)]
    features3 =["Chroma Vector_%i" % i for i in range(0,12)]
    features_names = ["%s" %  feature for feature in features1 + features2 + features3 +  ["Chroma Deviation"] ]
    print('%d channels, %d sampling rate\n' % (f.getnchannels(),
                                               f.getframerate()))
    # Set attributes
    device.setchannels(f.getnchannels())
    device.setrate(f.getframerate())
    # 8bit is unsigned in wav files
    if f.getsampwidth() == 1:
        device.setformat(alsaaudio.PCM_FORMAT_U8)
    # Otherwise we assume signed data, little endian
    elif f.getsampwidth() == 2:
        device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    elif f.getsampwidth() == 3:
        device.setformat(alsaaudio.PCM_FORMAT_S24_3LE)
    elif f.getsampwidth() == 4:
        device.setformat(alsaaudio.PCM_FORMAT_S32_LE)
    else:
        raise ValueError('Unsupported format')

    periodsize = int(f.getframerate()) #6ms windows

    device.setperiodsize(periodsize)

    data = f.readframes(periodsize)
    output = list()
    while data:
        # Read data from stdin
        data = np.abs(np.fromstring(data, dtype='int16'))
        stWin = round(f.getframerate() * 0.06)
        stats = aF.stFeatureExtraction(data, f.getframerate(), stWin / 4, stWin)
        myDF = pd.DataFrame(stats.transpose(), columns=features_names)
        if ( (myDF['MFCC_1'] > 1.5).any()):
            device.write(data)
        data = f.readframes(periodsize)
    return output

device = alsaaudio.PCM(device="default")
output = play(device, f)

import pandas as pd
# pd.DataFrame(output).plot()


my_data = aF.stFeatureExtraction(output, f.getframerate(), stWin / 4, stWin)


# myDF = pd.DataFrame(my_data.transpose(), columns=features_names)

myDF.plot()