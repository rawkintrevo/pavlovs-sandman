

### Install `pyalsaaudio`

```bash
sudo apt-get install python-dev
sudo apt-get install python3-dev
sudo apt-get install libasound2-dev
sudo pip install pyalsaaudio
```


### Plan

1. Record when ever there is a loud noise.
1. Continue recording for some duration
1. If this goes on too long (say 30 seconds?) classify as snoring.
1. Once loud noise has been quited, dump sound files to disk.
1. "Execute Action" if this threshold is tripped.

