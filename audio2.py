import alsaaudio, wave, numpy, os
import copy
from datetime import datetime, timedelta
import pandas as pd
from time import sleep
import alsaaudio, wave, numpy as np, pandas as pd, os
import pyAudioAnalysis.audioFeatureExtraction as aF
import pyAudioAnalysis.audioTrainTest as aT
from sklearn.externals import joblib

## TODO
# 1. Adaptive to background noise / threshold
# 2. Get Raw Audio / VOlume Hx Dumped to correct place (base_dir/date/time.ext)
# 3. Hack Shocker

class AudioHandler:
    def __init__(self):
        self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
        self.inp.setchannels(1)
        self.framerate = 44100
        self.inp.setrate(self.framerate)
        self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self.inp.setperiodsize(441)

        self.secondsUntilExecuteActionThreshold = 10
        self.lastExecuteActionTime = datetime.now()
        self.minSecondsBetweenExecuteAction = 10


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
        self.features_names = ["%s" %  feature for feature in features1 + features2 + features3 +  ["Chroma Deviation"] ]
        self.features_names2 = features_names = ["%s %s" % (m, f) for m in ["Mean", "Stdev"] for f in features1 + features2 + features3 +  ["Chroma Deviation"] ]
        self.stActiveBuffer = []
        self.mtActiveBuffer = []
        self.possibleSnoreActive = False
        self.clf = clf = joblib.load('svc-1-11-2018.pkl')

    def dumpData(self, epoch, score):
        assert isinstance(epoch, datetime)
        directory = "%i-%i-%i" % (epoch.year, epoch.month, epoch.day)
        if not os.path.exists("recordings/" + directory):
            os.makedirs("recordings/" + directory)
        if not os.path.exists("volume_data/" + directory):
            os.makedirs("volume_data/" + directory)
        # create directory in 'recordings' for date
        filename = "%s/%i-%i-%i_%.4f" % (directory, epoch.hour, epoch.minute, epoch.second, score)
        w = wave.open("recordings/" +filename + ".wav", 'w')
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        for data in self.mtActiveBuffer:
            w.writeframes(data)
        # pd.DataFrame.from_records(self.volume, index="timestamp").to_csv("volume_data/" +filename + ".csv")
        self.mtActiveBuffer = []
        self.volume = list()

    def analyzeSnore(self):
        stWin = round(self.framerate * 0.06)
        mtWin = 0.1
        mtStep = 0.1
        print("decomposing features", len(self.stActiveBuffer), self.framerate, mtWin, mtStep, 0.2, 0.2)
        Fs = self.framerate
        stats, foo = aF.mtFeatureExtraction(np.concatenate(self.stActiveBuffer), self.framerate,
                                       round(mtWin * Fs), round(mtStep * Fs),
                                            round(Fs * aT.shortTermWindow), round(Fs * aT.shortTermStep))
        # print(stats)
        myDF = pd.DataFrame(stats.transpose(), columns=self.features_names2)
        pred = self.clf.predict(myDF)
        return float(sum(pred)) / len(pred)

    def run(self):
        #self.setThreshold()
        # epoch = datetime.now()
        l, data = self.inp.read()
        data = numpy.fromstring(data, dtype='int16')
        self.activeFrame = data
        i = 0
        while True:
            i += 1
            # Read data from stdin
            l, data = self.inp.read()
            data = np.fromstring(data, dtype='int16')
            self.stActiveBuffer.append(data)
            # self.activeBuffer.append(data)
            if i % 50 == 0:
                # print("half second % i " % (i), len(self.activeBuffer))
                stWin = round(self.framerate * 0.06)
                stats = aF.stFeatureExtraction(np.concatenate(self.stActiveBuffer), self.framerate,
                                               stWin / 4, stWin)
                myDF = pd.DataFrame(stats.transpose(), columns=self.features_names)
                print(myDF['MFCC_1'].max())
                if ( (myDF['MFCC_1'] > 1.5).any()):
                    print("possible snore")
                    if self.possibleSnoreActive == False:
                        self.mtActiveBuffer = copy.deepcopy(self.stActiveBuffer)
                        self.possibleSnoreActive = True
                    else:
                        temp =  copy.deepcopy(self.stActiveBuffer)
                        self.mtActiveBuffer = self.mtActiveBuffer + temp
                        print(len(self.mtActiveBuffer))
                elif self.possibleSnoreActive: # but not MFCC_1 > 1.5
                    print("dumping data")
                    score = self.analyzeSnore()
                    self.dumpData(datetime.now(), score)
                    self.possibleSnoreActive = False
            if len(self.stActiveBuffer) < 98:
                ## Warm up
                continue
            else:
                foo = self.stActiveBuffer.pop(0)
            # print("eol:" , len(self.stActiveBuffer))

                # print(len(data))



    def executeAction(self):
        print("~~~~~~~~~~~~~~~~~ Shut up ALready!!!!!!!!!!!!!!!!!!!!!!!!!")

