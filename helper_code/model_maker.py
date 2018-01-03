import pandas as pd

import pyAudioAnalysis.audioFeatureExtraction as aF
import pyAudioAnalysis.audioTrainTest as aT

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

def prepDfForTraining(df):
    return df.loc[:, df.columns != 'label']

def trainModel(classifier, training_dataFrame):
    df = prepDfForTraining(training_dataFrame)

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', classifier)
    ])

    PCA_FEATURES_OPTIONS = [4, 8, 16, 32]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            'reduce_dim': [PCA()],
            'reduce_dim__n_components': PCA_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS,
            'classify__probability' : [True]
        }
    ]

    grid = GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=param_grid)
    grid.fit(df, training_dataFrame['label'])
    print("Model Fit, making sketch predictions")
    training_data2 = predictModel(df, grid, training_dataFrame['label'].max() + 1)
    return (training_data2.join(training_dataFrame['label']), grid)

def predictModel(df, grid, classes):
    df2 = df.copy()
    df2['preds'] = grid.predict(df)
    probaDF = pd.DataFrame(grid.predict_proba(df), columns= ["proba_%i" % i for i in range(0, classes)],
                            index= df.index)
    return df2.join(probaDF)


def createDataFramesFromFolders(folders, mtWin = 0.1, mtStep = 0.1):
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
    data_frames = list()
    l = 0
    for f in folders:
        (allMtFeatures, wavFilesList) = aF.dirWavFeatureExtraction(f,
                                                                   mtWin,
                                                                   mtStep,
                                                                   aT.shortTermWindow,
                                                                   aT.shortTermStep)

        sampleDF = pd.DataFrame(allMtFeatures, columns = features_names)
        sampleDF['label'] = l
        sampleDF.index = wavFilesList
        data_frames.append(sampleDF)
        l += 1
    return pd.concat(data_frames)