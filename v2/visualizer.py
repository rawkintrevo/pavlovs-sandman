import pyaudio
import time
import pylab
import numpy as np


from datetime import datetime
# https://github.com/ksingla025/pyAudioAnalysis3
from pyAudioAnalysis3.audioFeatureExtraction import stFeatureExtraction, mtFeatureExtraction
import pyAudioAnalysis3.audioTrainTest as aT

class SWHear(object):
    """
    The SWHear class is made to provide access to continuously recorded
    (and mathematically processed) microphone data.
    https://www.swharden.com/wp/2016-07-19-realtime-audio-visualization-in-python/
    """

    def __init__(self,device=None,startStreaming=True):
        """fire up the SWHear class."""
        print(" -- initializing SWHear")

        self.chunk = 4096 # number of data points to read at a time
        self.rate = 44100 # time resolution of the recording device (Hz)
        self.channels = 1
        self.format = pyaudio.paInt16
        # for tape recording (continuous "tape" of recent audio)
        self.tapeLength=3 #seconds
        self.tape=np.empty(self.rate*self.tapeLength)*np.nan

        self.feature_chunks = 12
        self.feature_tape = np.empty((self.tapeLength*self.feature_chunks, 34 ))*np.nan
        self.startTime = datetime.now()
        self.p=pyaudio.PyAudio() # start the PyAudio class
        if startStreaming:
            self.stream_start()

    ### LOWEST LEVEL AUDIO ACCESS
    # pure access to microphone and stream operations
    # keep math, plotting, FFT, etc out of here.

    def stream_read(self):
        """return values for a single chunk"""
        data = np.fromstring(self.stream.read(self.chunk),dtype=np.int16)
        #print(data)
        return data

    def stream_start(self):
        """connect to the audio device and start a stream"""
        print(" -- stream started")
        self.stream=self.p.open(format=pyaudio.paInt16,channels=1,
                                rate=self.rate,input=True,
                                frames_per_buffer=self.chunk)

    def stream_stop(self):
        """close the stream but keep the PyAudio instance alive."""
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
        print(" -- stream CLOSED")

    def close(self):
        """gently detach from things."""
        self.stream_stop()
        self.p.terminate()

    ### TAPE METHODS
    # tape is like a circular magnetic ribbon of tape that's continously
    # recorded and recorded over in a loop. self.tape contains this data.
    # the newest data is always at the end. Don't modify data on the type,
    # but rather do math on it (like FFT) as you read from it.

    def tape_add(self):
        """add a single chunk to the tape."""
        self.tape[:-self.chunk]=self.tape[self.chunk:]
        self.tape[-self.chunk:]=self.stream_read()
        self.feature_tape[:-self.feature_chunks]=self.feature_tape[self.feature_chunks:]



    def tape_flush(self):
        """completely fill tape with new data."""
        readsInTape=int(self.rate*self.tapeLength/self.chunk)
        print(" -- flushing %d s tape with %dx%.2f ms reads"% \
              (self.tapeLength,readsInTape,self.chunk/self.rate))
        for i in range(readsInTape):
            self.tape_add()

    def tape_forever(self,plotSec=.25):
        t1=0
        try:
            while True:
                self.tape_add()
                if (time.time()-t1)>plotSec:
                    t1=time.time()
                    self.tape_plot()
        except:
            print(" ~~ exception (keyboard?)")
            import traceback
            traceback.print_exc()
            return

    def tape_plot(self,saveAs="03.png"):
        """plot what's in the tape."""
        # print(self.feature_tape)
        # print(self.tape)
        #pylab.plot(np.arange(len(self.tape))/self.rate,self.tape)
        if int((datetime.now() - self.startTime).total_seconds()) % 3 == 0:
            print("data dump")
            self.dumpData()
        plot_data = np.transpose(self.feature_tape)
        for i in range(0,34):
            row = plot_data[i,:]
            pylab.plot(np.arange(len(row))/self.feature_chunks, row)
        # pylab.axis([0,plot_data,-2**16/2,2**16/2])
        pylab.axis([0, self.tapeLength, -5,5])
        if saveAs:
            t1=time.time()
            pylab.savefig(saveAs,dpi=50)
            #print("plotting saving took %.02f ms"%((time.time()-t1)*1000))
        else:
            pylab.show()
            # print() #good for IPython
        pylab.close('all')

    def extractFeatures(self, data):
        """
        Extract Audio Features Win/Step in samples if equal its a tumbling window
        :param data: the signal
        :return:  a numpy matrix of 34 rows and N columns, where N is the number of short-term
            frames that fit into the input audio recording
        """
        t1=time.time()
        windowSize = len(data)/self.feature_chunks
        # windowSize = aT.shortTermWindow
        output = stFeatureExtraction(data, Fs= self.rate, Win= windowSize, Step= windowSize)
        # output = mtFeatureExtraction(data, Fs= self.rate,
        #                              mtWin= len(data)/self.tapeLength,
        #                              mtStep= len(data/self.tapeLength),
        #                              stWin= 9, #aT.shortTermWindow,
        #                              stStep= 9) #aT.shortTermStep)
        # print("feature extraction took %.02f ms"%((time.time()-t1)*1000))
        return output

    def dumpData(self):
        import os
        from datetime import datetime
        import wave
        epoch = datetime.now()
        directory = "%i-%i-%i" % (epoch.year, epoch.month, epoch.day)
        if not os.path.exists("recordings/" + directory):
            os.makedirs("recordings/" + directory)
        if not os.path.exists("volume_data/" + directory):
            os.makedirs("volume_data/" + directory)
        # create directory in 'recordings' for date
        filename = "%s/%i-%i-%i" % (directory, epoch.hour, epoch.minute, epoch.second)
        w = wave.open("recordings/" +filename + ".wav", 'wb')
        w.setnchannels(self.channels)
        w.setsampwidth(self.p.get_sample_size(self.format))
        w.setframerate(self.rate)
        # for data in self.tape:
        w.writeframes(b"".join(self.tape))
        w.close()

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


if __name__=="__main__":
    ear=SWHear()
    ear.tape_forever()
    ear.close()
    print("DONE")