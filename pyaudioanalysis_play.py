
import pyAudioAnalysis.audioFeatureExtraction as aF
import pyAudioAnalysis.audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import pandas as pd


snoreRecordingsPath = "training_data/recordings/snore"
noSnoreRecordingsPath = "training_data/recordings/non-snore"
snoreExample = "4-46-14.wav"

# [Fs, x] = audioBasicIO.readAudioFile(unlabeledRecordingsPath + "/" + snoreExample);
# F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel('ZCR');
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel('Energy'); plt.show()

mtWin = 0.1
mtStep = 0.1

# ## You're getting 68 features, either averaged over file, or in mT windows
# (ltAverageFeatures, wavFilesList) = aF.dirWavFeatureExtraction(snoreRecordingsPath,
#                                     mtWin,
#                                     mtStep,
#                                     aT.shortTermWindow,
#                                     aT.shortTermStep)

## Load some good snoreing examples
(allMtFeatures, signalIndices, wavFilesList) = aF.dirWavFeatureExtractionNoAveraging(snoreRecordingsPath,
                                                        mtWin,
                                                        mtStep,
                                                        aT.shortTermWindow,
                                                        aT.shortTermStep)

print("%i windows found" % allMtFeatures.shape[0])


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

noSnoreSampleDF = pd.DataFrame(allMtFeatures, columns = features_names)
snoreSampleDF = pd.DataFrame(allMtFeatures, columns = features_names)
(snoreSampleDF - noSnoreSampleDF).plot()
(snoreSampleDF - noSnoreSampleDF).std().sort_values()
# MFCC_0 std dev of mean and stdev
# maybe stddev of mean of Entropy of energy and mfcc 1,2,4,3,7...

## Repeat experiment

(allMtFeatures, signalIndices, wavFilesList) = aF.dirWavFeatureExtractionNoAveraging(snoreRecordingsPath + "/2017-11-28",
                                                                                     mtWin,
                                                                                     mtStep,
                                                                                     aT.shortTermWindow,
                                                                                     aT.shortTermStep)

snoreSampleDF = pd.DataFrame(allMtFeatures, columns = features_names)

(allMtFeatures, signalIndices, wavFilesList) = aF.dirWavFeatureExtractionNoAveraging(noSnoreRecordingsPath + "/2017-11-28",
                                                                                     mtWin,
                                                                                     mtStep,
                                                                                     aT.shortTermWindow,
                                                                                     aT.shortTermStep)

noSnoreSampleDF = pd.DataFrame(allMtFeatures, columns = features_names)
(snoreSampleDF - noSnoreSampleDF).plot()
(snoreSampleDF - noSnoreSampleDF).std().sort_values()

# Same top contenders + Mean of Spectral Entropy

# I believe these signals DO contain information that will be useful to a snore detector

(allMtFeatures, wavFilesList) = aF.dirWavFeatureExtraction(snoreRecordingsPath,
                                                                                     mtWin,
                                                                                     mtStep,
                                                                                     aT.shortTermWindow,
                                                                                     aT.shortTermStep)

snoreSampleDF = pd.DataFrame(allMtFeatures, columns = features_names)
snoreSampleDF['wavFile'] = wavFilesList

(allMtFeatures, wavFilesList) = aF.dirWavFeatureExtraction(noSnoreRecordingsPath,
                                                                                     mtWin,
                                                                                     mtStep,
                                                                                     aT.shortTermWindow,
                                                                                     aT.shortTermStep)

noSnoreSampleDF = pd.DataFrame(allMtFeatures, columns = features_names)
noSnoreSampleDF['wavFile'] = wavFilesList

(snoreSampleDF.drop('wavFile', 1) - noSnoreSampleDF.drop('wavFile', 1)).plot()
(snoreSampleDF.drop('wavFile', 1) - noSnoreSampleDF.drop('wavFile', 1)).std().sort_values()


snoreSampleDF['label'] = 1
noSnoreSampleDF['label'] = 0

import pandas as pd
training_data = pd.concat([snoreSampleDF, noSnoreSampleDF])

independentVars = list((snoreSampleDF.drop('wavFile', 1) - noSnoreSampleDF.drop('wavFile', 1)).std().sort_values()[-4:].index)
dependentVars = ['label']


from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib

X_train, X_test, y_train, y_test = train_test_split(training_data[independentVars], training_data[dependentVars])

parameters = {'C':[0.1, 10.0],
              "probability" : [True]}

classifier = SVC()
clf = GridSearchCV(classifier, parameters)
c, r = y_train.shape
clf.fit(X_train, y_train.__array__().reshape(c,))

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

training_data['pred'] = clf.predict(training_data[independentVars])

training_data[(training_data['pred'] == 1) & (training_data['label'] == 0)]['wavFile']


from sklearn.decomposition import PCA

pca_data = PCA(n_components= 4).fit_transform(training_data[independentVars])

X_train, X_test, y_train, y_test = train_test_split(pca_data, training_data[dependentVars])

parameters = {'C':[0.1, 10.0],
              "probability" : [True]}

classifier = SVC()
clf = GridSearchCV(classifier, parameters)
c, r = y_train.shape
clf.fit(X_train, y_train.__array__().reshape(c,))

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

training_data['preds2'] = clf.predict(pca_data)

training_data[(training_data['preds2'] == 1) & (training_data['label'] == 0)]['wavFile']

probaDF = pd.DataFrame(clf.predict_proba(pca_data), columns= ["proba_0", "proba_1"], index= training_data['wavFile'])
training_data.index= training_data['wavFile']

training_data2 = training_data.join(probaDF)
training_data2[(training_data2['preds2'] == 1) & (training_data2['label'] == 0)][['wavFile', "proba_0", "proba_1", "pred"]]


from helper_code.model_maker import createDataFramesFromFolders, trainModel, prepDfForTraining, predictModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

clf = SVC()

otherRecordingsPath1 = "training_data/recordings/other noises"
otherRecordingsPath2 = "training_data/recordings/heavybreathing"
tdf = createDataFramesFromFolders([snoreRecordingsPath,
                                   noSnoreRecordingsPath,
                                   otherRecordingsPath1,
                                   otherRecordingsPath2])
(tdf2, grid) = trainModel(clf, tdf)

tdf2[(tdf2['preds']==0) & (tdf2['label'] > 0)][["proba_%i" % i for i in range(0,4)] + ["preds"]]

unlabeledPath = "training_data/recordings/unlabeled/2017-11-28"
unlabeledDf = createDataFramesFromFolders([unlabeledPath])
unlabeledDf2 = prepDfForTraining(unlabeledDf)
tdf3 = predictModel(unlabeledDf2, grid, 4)

tdf3[(tdf3['preds']==0)][["proba_0"]].index




