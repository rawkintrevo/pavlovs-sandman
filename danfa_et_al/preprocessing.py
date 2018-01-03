
import numpy as np

from scipy.stats import signaltonoise

from helper_code.audio import AudioHandler

class Preprocessing(AudioHandler):
    def __init__(self):
        pass

    def wienerFilter(self, sample):
        """

        :param sample: array_like
        :return:
        """
        weinerFiltered = signaltonoise(sample) / (1+ signaltonoise(sample))

    def run(self):
        """
        Main Run Method
        :return:
        """
        # self.volumeThreshold = 1400
        while True:
            # epoch = datetime.now()
            l, dataStr = self.inp.read()
            data = np.abs(np.fromstring(dataStr, dtype='int16'))
            filteredData = self.weinerFiltered(data)

            #
            # if vol > self.volumeThreshold:
            #     print("Loud noises!!", vol)
            #     self.lastLoudNoise = epoch
            #     if self.recordingActive:
            #         if (epoch - self.activeRecordingStartedAt).seconds > 10:
            #             if (epoch - self.lastExecuteActionTime).seconds > self.minSecondsBetweenExecuteAction:
            #                 self.lastExecuteActionTime = epoch
            #                 self.executeAction()
            #             else:
            #                 print("you're testin my nerves buddy!")
            #     else:
            #         self.recordingActive = True
            #         self.activeRecordingStartedAt = epoch
            # if self.recordingActive:
            #     self.volume.append({"timestamp":epoch, "value": vol})
            #     self.rawData.append(data)
            #     if (epoch - self.lastLoudNoise).seconds > self.secondsUntilExecuteActionThreshold:
            #         print("Ahh finally some peace and quite!")
            #         self.dumpData(epoch)
            #         ## dump volumes
            #         self.recordingActive = False


