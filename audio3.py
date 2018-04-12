import pyaudio
import time
import pylab
import numpy as np
from datetime import datetime

import shock_collar.shock_collar



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
        self.tapeLength=5 #seconds
        self.lastShock = datetime.now()
        self.tape=np.empty(self.rate*self.tapeLength)*np.nan
        self.filterTape1 = np.empty(self.rate*self.tapeLength)*np.nan
        self.shockLevel = 1
        # self.feature_chunks = 12
        # self.feature_tape = np.empty((self.tapeLength*self.feature_chunks, 34 ))*np.nan
        print("initializing collar iface")
        self.sc = shock_collar.shock_collar.Controller()
        print("collar iface init completed")
        self.p=pyaudio.PyAudio() # start the PyAudio class
        if startStreaming:
            self.stream_start()
        self.reset_shock_level_every_n_seconds = 15
        self.log_file = "/home/rawkintrevo/gits/pavlovs-sandman/logs/records.csv"
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
        self.stream=self.p.open(format=self.format,channels=self.channels,
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
        # self.filterTape1[:-self.chunk]= self.filterTape1[self.chunk:]
        # self.filterTape1[-self.chunk:]= self.filterFreqs(self.tape[-self.chunk:], \
        #                                                                         lowpass=1500, \
        #                                                                         highpass=2000)[0]


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
                    # print("monitoring")
                    t1=time.time()
                    # self.tape_plot()
                    self.monitor()
        except:
            print(" ~~ exception (keyboard?)")
            import traceback
            traceback.print_exc()
            return

    def monitor(self):
        lb, hb = self.snoreOrNo(self.tape)
        if lb > 0 or hb > 0:
            print(( datetime.now() - self.lastShock).total_seconds())
            print("tape length: ", len(self.tape) / self.rate)
            if ((datetime.now() - self.lastShock).total_seconds()) > self.tapeLength:
                print(datetime.now().strftime("%Y-%m-%d - %H:%M:%S"),
                      " : snore detected, writing to disk")
                self.dumpData()
                try:
                    if (datetime.now() - self.lastShock).seconds < self.reset_shock_level_every_n_seconds:
                        self.shockLevel = min(self.shockLevel + 1, 10)
                    else:
                        self.shockLevel = 1
                    print("Shock level: %i" % self.shockLevel)
                    with open(self.log_file, "a") as f:
                        # datetime, shock level
                        f.write("%s,%i,1\n" % (datetime.now().strftime("%Y-%m-%d - %H:%M:%S"), self.shockLevel) )
                    self.sc.tone()
                    self.sc.shock(self.shockLevel)
                except:
                    print("Error communicating with device")
                    with open(self.log_file, "a") as f:
                        # datetime, shock level
                        f.write("%s,%i,0\n" % (datetime.now().strftime("%Y-%m-%d - %H:%M:%S"), self.shockLevel) )
                # self.sc.buzz()
                self.lastShock = datetime.now()

    def tape_plot(self,saveAs="v2/03.png"):
        """plot what's in the tape."""
        # print(self.feature_tape)
        # print(self.tape)
        lb, hb = self.snoreOrNo(self.filterTape1)
        pylab.plot(np.arange(len(self.filterTape1))/self.rate,self.filterTape1)
        pylab.xlabel("lb: %i ; hb: %i" % (lb, hb))
        # pylab.axis([0,plot_data,-2**16/2,2**16/2])
        if saveAs:
            t1=time.time()
            pylab.savefig(saveAs,dpi=50)
            print("plotting saving took %.02f ms"%((time.time()-t1)*1000))
        else:
            pylab.show()
            # print() #good for IPython
        pylab.close('all')

    def filterFreqs(self, data, lowpass=0, highpass=22050, filters= [], stereo= False):
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


    def durations(self, data_filtered):
        # look at max over 10/th second blocks
        data_mean = np.mean(np.abs(data_filtered))
        data_std = np.std(np.abs(data_filtered))
        thresh = data_mean + 2.0*data_std
        snore = False
        activeSeq = []
        output = []
        for i in range(0, len(data_filtered), 4410):
            val = np.max(data_filtered[i: i+4410])
            # print(val/thresh)
            if val / thresh > 1:
                if snore == False:
                    # print("noSnore len: ", len(activeSeq)/10)
                    output.append(("ns", float(len(activeSeq))/10))
                    snore = True
                    activeSeq = [1]
                else:
                    activeSeq.append(1)
            if val / thresh < 1:
                if snore == True:
                    # print("snore len: ", len(activeSeq)/10)
                    output.append(("s", float(len(activeSeq))/10))
                    snore = False
                    activeSeq = [0]
                else:
                    activeSeq.append(0)
        return output

        ## Any snore of 0.1 should be turned into a "noSnore" and then merge the adjacent "noSnores"
    def parseDurrationsForSnores(self, durrs):
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

    def snoreOrNo(self, data):
        data_filtered1, fft = self.filterFreqs(data, lowpass= 1500, highpass=2000)
        data_filtered2, fft = self.filterFreqs(data, lowpass= 8000, highpass=15000)
        durrs1 = self.durations(data_filtered1)
        # print("Durrs1: ", durrs1)
        c1 = self.parseDurrationsForSnores(durrs1)
        durrs2 = self.durations(data_filtered2)
        # print("Durrs2: ", durrs2)
        c2 = self.parseDurrationsForSnores(durrs2)
        # print(c1,c2)
        return c1, c2

    def dumpData(self):
        import os
        from datetime import datetime
        import scipy.io.wavfile
        epoch = datetime.now()
        directory = "%i-%i-%i" % (epoch.year, epoch.month, epoch.day)
        if not os.path.exists("recordings/" + directory):
            os.makedirs("recordings/" + directory)
        if not os.path.exists("volume_data/" + directory):
            os.makedirs("volume_data/" + directory)
        # create directory in 'recordings' for date
        filename = "%s/%i-%i-%i" % (directory, epoch.hour, epoch.minute, epoch.second)
        scipy.io.wavfile.write("recordings/" +filename + ".wav", self.rate, self.tape)

if __name__=="__main__":
    ear=SWHear()
    ear.tape_forever()
    ear.close()
    print("DONE")