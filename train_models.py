
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib

from helper_code.prepocessing import summarizeVolumeData

dateTarget = '2017-11-28'
snorePath = 'training_data/volume_data/snore/%s' % dateTarget
noSnorePath = 'training_data/volume_data/non-snore/%s' % dateTarget

snoreCsvs = os.listdir(snorePath)
nonSnoreCsvs = os.listdir(noSnorePath)

independentVars = ['meanDiff', 'stdDiff', 'ticksPerPeak']
dependentVars = ['label']
training_data = pd.concat([summarizeVolumeData(snoreCsvs, snorePath, 1),
                           summarizeVolumeData(nonSnoreCsvs, noSnorePath, 0)])[independentVars + dependentVars]

X_train, X_test, y_train, y_test = train_test_split(training_data[independentVars], training_data[dependentVars])

parameters = {'C':[0.1, 10.0],
              "probability" : [True]}

classifier = SVC()
clf = GridSearchCV(classifier, parameters, scoring= make_scorer(precision_score))
c, r = y_train.shape
clf.fit(X_train, y_train.__array__().reshape(c,))

fname = "data/trained_models/cheapSVC1.pkl"
print("saving: %s" % fname)

joblib.dump(clf.best_estimator_, fname)


from pyAudioAnalysis import audioTrainTest as aT


PAAsnorePath = 'training_data/pyaudioanalysis/snore/'
PAAnoSnorePath = 'training_data/pyaudioanalysis/non-snore/'

clf_names = [
    'svm_rbf',
    'svm',
    'knn',
    'gradientboosting',
    'randomforest',
    'extratrees'
]

clf = clf_names[1]
modelName = "data/trained_models/paaSVC1"
aT.featureAndTrain([PAAsnorePath, PAAnoSnorePath],
                   1.0, 1.0, # midTerm window / step
                   aT.shortTermWindow, aT.shortTermStep,  # shortTerm
                   clf,       # classifierName
                   modelName, # modelName
                   False) # beat calculation, music only
