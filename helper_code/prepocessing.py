
import numpy as np
import pandas as pd

from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema


def summarizeVolumeData(listOfCsvs, path= "", label = ""):
    """

    :param listOfCsvs:  list- csvs
    :param path:        string- path to where csvs reside
    :param label:       optional label to tack on
    :return: pd.DataFrame
    """
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
                        "label" : label,
                        "name"  : csv.replace(".csv", "")})
    return pd.DataFrame(results)
