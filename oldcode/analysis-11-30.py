import pandas as pd
import matplotlib.pyplot as plt

import os

dateTarget = '2017-11-28'
snorePath = 'training_data/volume_data/snore/%s' % dateTarget
noSnorePath = 'training_data/volume_data/non-snore/%s' % dateTarget

snoreCsvs = os.listdir(snorePath)
nonSnoreCsvs = os.listdir(noSnorePath)

## Quick Inspection
howManyPlots = 20
for i in range(howManyPlots):
    df = pd.DataFrame \
        .from_csv("%s/%s" % (noSnorePath, nonSnoreCsvs[i]))

for i in range(howManyPlots):
    df = pd.DataFrame \
        .from_csv("%s/%s" % (snorePath, snoreCsvs[i])) \
        .plot(title = "Snore @ %s" % snoreCsvs[i].replace(".csv", ""))


########## Smells like ARIMA
# good refresher + python code : https://datascience.ibm.com/exchange/public/entry/view/815137c868b916821dec777bdc23013c

# ## Fuck statsmodels
# import numpy as np
# from statsmodels.tsa.arima_model import ARIMA
# p = 80
# d = 1
# q = 15
# arima = ARIMA(np.log(df['value']), order=(p, d, q))
# res = arima.fit( start_ar_lags=p)

####################################################################################################
from scipy.ndimage.filters import gaussian_filter
import numpy as np

data = pd.DataFrame(np.diff(df['value']), columns=['value'])
for i in range(75, 86):
    data['y'] = gaussian_filter(data['value'], 10)
    data['lag_%i' % i] = data['y'].shift(i)

from scipy.signal import argrelextrema
localMax = argrelextrema(data['y'].values, np.greater)

data = data.dropna()


## doesn't give param estimates
from sklearn.linear_model import LinearRegression

from statsmodels.formula.api import ols
model = ols("y ~ " + " + ".join(['lag_%i' % i for i in range(75,86)]), data).fit()

model


for i in range(howManyPlots):
for r in range(5, 50, 10):
    noisyDf = pd.DataFrame \
        .from_csv("%s/%s" % (snorePath, "1-8-18.csv"))
    noisyDf['gausFilter'] = gaussian_filter(noisyDf['value'],   r)
    noisyDf = noisyDf.reset_index()
    noisyDf['gausFilter'].plot(title = "Smoothed Snores Sample")


## Do a gaus filter of 20 count peeks per total time
results = list()
from scipy.signal import argrelextrema
import numpy as np

def summarizeVolumeData(listOfCsvs, path= "", label = ""):
    results = list()
    for csv in listOfCsvs:
        noisyDf = pd.DataFrame \
            .from_csv("%s/%s" % (path, csv))
        x = gaussian_filter(noisyDf['value'], 10)
        localMax = argrelextrema(x, np.greater)
        results.append({"size" : x.size ,
                        "nLocalMax" : localMax[0].size,
                        "meanDiff" : np.diff(localMax[0]).mean(),
                        "ticksPerPeak" : x.size / localMax[0].size,
                       "stdDiff" : np.diff(localMax[0]).std(),
                       "label" : label})
    return pd.DataFrame(results)

snorePath = 'training_data/volume_data/snore/%s' % dateTarget
noSnorePath = 'training_data/volume_data/non-snore/%s' % dateTarget

snoreCsvs = os.listdir(snorePath)
nonSnoreCsvs = os.listdir(noSnorePath)

print(summarizeVolumeData(snoreCsvs, snorePath, 1).mean())
print(summarizeVolumeData(nonSnoreCsvs, noSnorePath, 0).mean())

print(summarizeVolumeData(snoreCsvs, snorePath, 1).std())
print(summarizeVolumeData(nonSnoreCsvs, noSnorePath, 0).std())


independentVars = ['meanDiff', 'stdDiff', 'ticksPerPeak']
dependentVars = ['label']
training_data = pd.concat([summarizeVolumeData(snoreCsvs, snorePath, 1),
           summarizeVolumeData(nonSnoreCsvs, noSnorePath, 0)])[independentVars + dependentVars]


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

X_train, X_test, y_train, y_test = train_test_split(training_data[independentVars], training_data[dependentVars])

classifier = LogisticRegression()
y_pred = classifier.fit(X_train, y_train).predict(X_test)

class_names = ['non-snore', 'snore']
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

### lots of false positives... das ist nicht so gut

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, make_scorer

parameters = {'solver': ('liblinear',
                         'newton-cg',
                         'lbfgs',
                         'sag' ), 'C':[0.1, 100.0]}

#classifier = LogisticRegression()
del parameters['solver']
classifier = SVC()
clf = GridSearchCV(classifier, parameters, scoring= make_scorer(precision_score))
c, r = y_train.shape
clf.fit(X_train, y_train.__array__().reshape(c,))

y_pred = clf.best_estimator_.predict(X_test)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

## SVC was "twice as good" as Logistic Regression ( 3 FP instead of 6 )
## Next Time: Check all False Positives-
## Maybe some misclasses? Listen to Sound, Look at Charts, reclass as appropriate.
## Maybe other ideas and pipeline - e.g. If LR is (+) Run it over a second test
## having trouble triaining- no predicted samples...
