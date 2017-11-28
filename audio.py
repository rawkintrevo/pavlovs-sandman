import alsaaudio, wave, numpy, os
from datetime import datetime, timedelta
import pandas as pd
from time import sleep

## TODO
# 1. Adaptive to background noise / threshold
# 2. Get Raw Audio / VOlume Hx Dumped to correct place (base_dir/date/time.ext)
# 3. Hack Shocker

class AudioHandler:
    def __init__(self):
        self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
        self.inp.setchannels(1)
        self.inp.setrate(44100)
        self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self.inp.setperiodsize(1024)
        self.rawData = list()
        self.volume = list()
        self.volumeThreshold = 15.0
        self.recordingActive = False
        self.activeRecordingStartedAt = datetime.now()
        self.lastLoudNoise =  datetime.now()
        self.secondsUntilExecuteActionThreshold = 10
        self.lastExecuteActionTime = datetime.now()
        self.minSecondsBetweenExecuteAction = 10
        self.warmUpMode = True

    def dumpData(self, epoch):
        assert isinstance(epoch, datetime)
        directory = "%i-%i-%i" % (epoch.year, epoch.month, epoch.day)
        if not os.path.exists("recordings/" + directory):
            os.makedirs("recordings/" + directory)
        if not os.path.exists("volume_data/" + directory):
            os.makedirs("volume_data/" + directory)
        # create directory in 'recordings' for date
        filename = "%s/%i-%i-%i" % (directory, epoch.hour, epoch.minute, epoch.second)
        w = wave.open("recordings/" +filename + ".wav", 'w')
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        for data in self.rawData:
            w.writeframes(data)
        pd.DataFrame.from_records(self.volume, index="timestamp").to_csv("volume_data/" +filename + ".csv")
        self.rawData = list()
        self.volume = list()

    def run(self):
        startTime = datetime.now()
        print("warming up")
        while self.warmUpMode:
            l, data = self.inp.read()
            vol = numpy.abs(numpy.fromstring(data, dtype='int16')).mean()
            self.volume.append(vol)
            if (datetime.now() - startTime).seconds > 10:
                self.warmUpMode = False
                self.volumeThreshold = numpy.max(self.volume) + numpy.std(self.volume)
                print(numpy.mean(self.volume), numpy.std(self.volume), numpy.max(self.volume))
        self.volume = list()
        print("warm up complete. threshold is %f" % self.volumeThreshold)
        while True:
            epoch = datetime.now()
            l, data = self.inp.read()
            vol = numpy.abs(numpy.fromstring(data, dtype='int16')).mean()
            if vol > self.volumeThreshold:
                print("Loud noises!!", vol)
                self.lastLoudNoise = epoch
                if self.recordingActive:
                    if (epoch - self.activeRecordingStartedAt).seconds > 10:
                        if (epoch - self.lastExecuteActionTime).seconds > self.minSecondsBetweenExecuteAction:
                            self.lastExecuteActionTime = epoch
                            self.executeAction()
                        else:
                            print("you're testin my nerves buddy!")
                else:
                    self.recordingActive = True
                    self.activeRecordingStartedAt = epoch
            if self.recordingActive:
                self.volume.append({"timestamp":epoch, "value": vol})
                self.rawData.append(data)
                if (epoch - self.lastLoudNoise).seconds > self.secondsUntilExecuteActionThreshold:
                    print("Ahh finally some peace and quite!")
                    self.dumpData(epoch)
                    ## dump volumes
                    self.recordingActive = False
                    # reset everything

    def executeAction(self):
        print("~~~~~~~~~~~~~~~~~ Shut up ALready!!!!!!!!!!!!!!!!!!!!!!!!!")


snore_detector = AudioHandler()
snore_detector.run()