import alsaaudio, wave, numpy as np, pandas as pd, os
import pyAudioAnalysis.audioFeatureExtraction as aF


snore_target_dirs = [("training_data/recordings/snore", "psnore"),
                     ("training_data/recordings/mixed", "psnore"),
                     ("training_data/recordings/non-snore", "other"),
                     ("training_data/recordings/heavybreathing", "other"),
                     ("training_data/recordings/other noises", "other")]


def parseDir(input_dir, output_dir, label):
    for fname in os.listdir(input_dir):
        parseWavFile(output_dir, label, fname, input_dir)


def parseWavFile(output_dir, label, fname, input_dir):
    f = wave.open('%s/%s' % (input_dir, fname)) #'training_data/recordings/snore/1-6-34.wav')
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
    periodsize = int(f.getframerate()) #6ms windows
    fname_base = "recordings/" +output_dir + '/' + label + "/" + fname + "-"
    data = f.readframes(periodsize)
    output = list()
    i = 0
    while data:
        # Read data from stdin
        data = np.abs(np.fromstring(data, dtype='int16'))
        stWin = round(f.getframerate() * 0.06)
        stats = aF.stFeatureExtraction(data, f.getframerate(), stWin / 4, stWin)
        myDF = pd.DataFrame(stats.transpose(), columns=features_names)
        if ( (myDF['MFCC_1'] > 1.5).any()):
            w = wave.open(fname_base + "%i.wav" % i, 'w')
            i += 1
            w.setnchannels(f.getnchannels)
            w.setsampwidth(f.getsampwidth)
            w.setframerate(f.getframerate)
            w.writeframes(data)
        data = f.readframes(periodsize)
    return output


for d, l in snore_target_dirs:
    parseDir(d, "clips", l)

