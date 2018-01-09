
import pyAudioAnalysis.audioFeatureExtraction as aF
import pyAudioAnalysis.audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import pandas as pd


snoreRecordingsPath = "training_data/recordings/clips/psnore"
noSnoreRecordinsPath = "training_data/recordings/clips/other"


mtWin = 0.1
mtStep = 0.1


## Load some good snoreing examples
(allMtFeaturesSnore, wavFilesListSnore) = aF.dirWavFeatureExtraction(snoreRecordingsPath,
                                                                                     mtWin,
                                                                                     mtStep,
                                                                                     aT.shortTermWindow,
                                                                                     aT.shortTermStep)
## Load some good NoSnoreing examples
(allMtFeaturesNoSnore,  wavFilesListNoSnore) = aF.dirWavFeatureExtraction(noSnoreRecordinsPath,
                                                                                                    mtWin,
                                                                                                    mtStep,
                                                                                                    aT.shortTermWindow,
                                                                                                    aT.shortTermStep)


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

features_names = ["%s %s" % (m, f) for m in ["Mean", "Stdev"] for f in features1 + features2 + features3 +  ["Chroma Deviation"] ]

noSnoreSampleDF = pd.DataFrame(allMtFeaturesNoSnore, columns = features_names)
snoreSampleDF = pd.DataFrame(allMtFeaturesSnore, columns = features_names)

noSnoreSampleDF['label'] = 0
snoreSampleDF['label'] = 1

noSnoreSampleDF['fileName'] = wavFilesListNoSnore
snoreSampleDF['fileName'] = wavFilesListSnore

(snoreSampleDF.drop('fileName', 1) - noSnoreSampleDF.drop('fileName', 1)).plot()


import pandas as pd
training_data = pd.concat([snoreSampleDF, noSnoreSampleDF])

independentVars = features_names
dependentVars = ['label']


from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

X_train, X_test, y_train, y_test = train_test_split(training_data[independentVars], training_data[dependentVars])

clfs = {'svc':{
    'classifier' : SVC(),
    'parameters' : {
              'C':[0.1, 10.0],
              "probability" : [True]}
            },
    "LogisticRegression": {
        'classifier' : LogisticRegression(),
        'parameters' : {
            'C': [0.1, 10.0]
        }
    }
}

output = list()
for k in clfs:
    clf = GridSearchCV(clfs[k]['classifier'], clfs[k]['parameters'])
    c, r = y_train.shape
    clf.fit(X_train, y_train.__array__().reshape(c,))
    y_pred = clf.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    output.append({'clf': k, 'fp' : fp, 'tp': tp, 'tn': tn, 'fn': fn})