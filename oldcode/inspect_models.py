
import os
import pandas as pd

from sklearn.externals import joblib

from helper_code.prepocessing import summarizeVolumeData

from pyAudioAnalysis import audioTrainTest as aT

dateTarget = '2017-11-28'
unlabeledCsvPath = 'training_data/volume_data/unlabeled/%s' % dateTarget

sk_model = "data/trained_models/cheapSVC1.pkl"
paa_model = "data/trained_models/paaSVC1"

sk_clf = joblib.load(sk_model)

csvDF = summarizeVolumeData(os.listdir(unlabeledCsvPath), unlabeledCsvPath)
independentVars = ['meanDiff', 'stdDiff', 'ticksPerPeak']

# Shows that a positive decision fn value => classified as snore
# import numpy as np
# import pandas as pd
## By inspection, clf.predict_proba is not returning sensical results- always ~.55 in favor- even though that isn't returned
# preds1 = np.vstack([sk_clf.decision_function(csvDF[independentVars]), sk_clf.predict(csvDF[independentVars])])
# pd.DataFrame(preds1.transpose()[0:20]).plot()


import numpy as np

def sigmoid(t):                          # Define the sigmoid function
    return (1/(1 + np.e**(-t)))

sk_df = pd.DataFrame(sk_clf.predict(csvDF[independentVars]), columns=['sk_dec_fn'])
sk_df['time'] = csvDF['name']
sk_df = sk_df.set_index("time")['sk_dec_fn'] #.apply(sigmoid)

modelName= "data/trained_models/paaSVC1"

recordingsPath = "training_data/recordings/unlabeled/2017-11-28"
recordings = os.listdir(recordingsPath)
output = list()
for csv in os.listdir(unlabeledCsvPath):
    if csv.replace("csv", "wav") in recordings:
        fname = recordingsPath + "/" + csv.replace("csv", "wav")
        output.append({
            "paa_snore_p" : aT.fileClassification(fname, modelName= modelName, modelType= 'svm')[1][0],
            "time" : csv.replace(".csv","")})

paa_df = pd.DataFrame(output).set_index("time")

paa_df.join(sk_df)[0:40]
#           paa_snore_p  sk_dec_fn
# time
# 7-18-5      0.385461   0.602433   # NO SNORE
# 6-16-37     0.235689   0.527842
# 5-33-37     0.208272   0.455065
# 3-51-9      0.843937   0.255997
# 2-52-42     0.335842   0.537919   # SNORE
# 3-55-13     0.754144   0.523265   # SNORE
# 6-21-49     0.221403   0.388087   # SNORE
# 5-42-8      0.071736   0.442013
# 6-13-28     0.317514   0.524025

# I intuitevely thinking dates are not alligning properly.