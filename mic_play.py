import alsaaudio, numpy as np

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
inp.setchannels(1)
framerate = 44100
inp.setrate(framerate)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(441)

while True:
    l, data = inp.read()
    data = np.fromstring(data, dtype='int16')
    print (len(data))
