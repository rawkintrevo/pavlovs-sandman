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

    print('%s: %d channels, %d sampling rate, %i sample width\n' % (fname,
                                                                    f.getnchannels(),
                                                f.getframerate(),
                                                f.getsampwidth()))
    # Set attributes
    periodsize = int(f.getframerate()) #6ms windows
    fname_base = "training_data/recordings/" +output_dir + '/' + label + "/" + fname + "-"
    output = list()
    i = 0
    possibleSnoreActive = True
    activeBuffer = []
    data = f.readframes(periodsize)
    lastFrame = False
    while data:
        # Read data from stdin
        data = np.abs(np.fromstring(data, dtype='int16'))
        stWin = round(f.getframerate() * 0.06)
        # if not lastFrame:
        print len(data)
        if len(data) < 44100:
            lastFrame = True
        else:
            stats = aF.stFeatureExtraction(data, f.getframerate(), stWin / 4, stWin)
            myDF = pd.DataFrame(stats.transpose(), columns=features_names)
        if ( (myDF['MFCC_1'] > 1.5).any()):
            print("%i possible snore" % i)
            possibleSnoreActive = True
            activeBuffer.append(data)
        if  possibleSnoreActive and (not ((myDF['MFCC_1'] > 1.5).any()) or lastFrame):
            output_file_name = fname_base.replace(".wav", "") + "%i.wav" % i
            print('writing: %s' % output_file_name)
            w = wave.open(output_file_name, 'w')
            w.setsampwidth(f.getsampwidth())
            w.setnchannels(f.getnchannels())
            w.setframerate(f.getframerate())
            [w.writeframes(frame) for frame in activeBuffer]
            activeBuffer = []
            possibleSnoreActive = False
        print(len(data))

        data = f.readframes(periodsize)
        i += 1
    return output


for d, l in snore_target_dirs:
    parseDir(d, "clips", l)

