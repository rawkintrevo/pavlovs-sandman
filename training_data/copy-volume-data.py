
import os, shutil

volume_dates = [f for f in  os.listdir("../volume_data") if os.path.isdir("../volume_data/%s" % f)]

def mkdirIfNotExists(path):
    if not os.path.exists(path):
        os.mkdir(path)

for d in volume_dates:
    volume_csvs = [f.replace(".csv","") for f in os.listdir("../volume_data/%s" % d)]
    for dir in ['snore', 'non-snore', 'unlabled']:
        mkdirIfNotExists("./volume_data/%s" % (dir))
        mkdirIfNotExists("./volume_data/%s/%s" % (dir, d))
        if not os.path.exists("recordings/%s/%s"  % (dir, d)): continue
        recordings = [f.replace(".wav", "") for f in os.listdir("recordings/%s/%s"  % (dir, d))]
        targets = list(set(volume_csvs).intersection(set(recordings)))
        for t in targets:
            shutil.copy("../volume_data/%s/%s.csv" % (d, t), "./volume_data/%s/%s/%s.csv" % (dir, d, t))
