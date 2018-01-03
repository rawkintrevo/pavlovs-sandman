## Python 2.7 - 3.0 is trash and I can't deal with it anymore

from pyAudioAnalysis import audioTrainTest as aT


snorePath = 'training_data/pyaudioanalysis/snore/'
noSnorePath = 'training_data/pyaudioanalysis/non-snore/'

clf_names = [
    'svm_rbf',
    'svm',
    'knn',
    'gradientboosting',
    'randomforest',
    'extratrees'
]

clf = clf_names[1]
modelName = "svmSMtemp"
aT.featureAndTrain([snorePath, noSnorePath],
                   1.0, 1.0, # midTerm window / step
                   aT.shortTermWindow, aT.shortTermStep,  # shortTerm
                   clf,       # classifierName
                   modelName, # modelName
                   False) # beat calculation, music only
print(clf)
# svm
    # C 	    PRE	    REC	    F1 	    PRE	    REC	    F1 	    ACC	    F1
    # 0.001 	89.8	62.9	74.0 	71.5	92.9	80.8 	77.9	77.4
    # 0.010 	92.2	75.8	83.2 	79.4	93.6	85.9 	84.7	84.5 	 best F1 	 best Acc
    # 0.500 	84.3	84.6	84.4 	84.5	84.2	84.4 	84.4	84.4
    # 1.000 	82.8	83.1	83.0 	83.1	82.8	82.9 	82.9	82.9
    # 5.000 	79.5	81.1	80.3 	80.7	79.1	79.9 	80.1	80.1
    # 10.000 	79.9	83.1	81.5 	82.4	79.1	80.7 	81.1	81.1
    # Confusion Matrix:
    # sno 	non
    # sno 	37.9 	12.1
    # non 	3.2 	46.8
    # Selected params: 0.01000

# knn:
    # snore			non-snore		OVERALL
    #   C 	    PRE	    REC	    F1 	    PRE	    REC    	F1 	    ACC	    F1
    # 1.000 	74.1	78.3	76.1 	77.0	72.6	74.7 	75.4	75.4
    # 3.000 	77.1	73.7	75.3 	74.8	78.1	76.4 	75.9	75.9
    # 5.000 	77.7	67.6	72.3 	71.3	80.7	75.7 	74.1	74.0
    # 7.000 	80.7	71.6	75.9 	74.5	82.9	78.4 	77.2	77.1
    # 9.000 	81.6	75.6	78.5 	77.2	83.0	80.0 	79.3	79.2
    # 11.000 	83.5	72.9	77.8 	75.9	85.6	80.5 	79.2	79.1
    # 13.000 	83.4	76.0	79.5 	78.0	84.9	81.3 	80.4	80.4 	 best F1 	 best Acc
    # 15.000 	83.5	74.2	78.6 	76.8	85.3	80.8 	79.8	79.7
    # Confusion Matrix:
    #       sno 	non
    # sno 	38.0 	12.0
    # non 	7.6 	42.4
    #
#
# Random Forrest
    # C 	    PRE 	REC	    F1  	PRE 	REC 	F1  	ACC	    F1
    # 10.000 	79.7	81.4	80.5 	81.0	79.2	80.1 	80.3	80.3
    # 25.000 	87.6	74.8	80.7 	78.0	89.4	83.3 	82.1	82.0
    # 50.000 	85.8	80.3	83.0 	81.5	86.7	84.0 	83.5	83.5
    # 100.000 	86.5	80.7	83.5 	81.9	87.4	84.6 	84.1	84.0
    # 200.000 	86.0	80.2	83.0 	81.5	86.9	84.1 	83.6	83.5
    # 500.000 	88.6	79.3	83.7 	81.3	89.8	85.3 	84.6	84.5 	 best F1 	 best Acc
    # Confusion Matrix:
    # sno 	non
    # sno 	39.7 	10.3
    # non 	5.1 	44.9

hard_guess = ['training_data/recordings/snore/2017-11-28/0-54-9.wav']

known_snore = ['training_data/recordings/unlabeled/2017-11-28/2-31-18.wav',
               'training_data/recordings/snore/2017-11-28/1-26-41.wav',
               'training_data/recordings/unlabeled/2017-11-28/2-41-32.wav']

not_a_snore= ['training_data/recordings/non-snore/2017-11-28/0-15-56.wav']

guess = aT.fileClassification( 'training_data/recordings/unlabeled/2017-11-28/3-35-21.wav',
                      modelName,
                      clf)

## Go through all unlabeleds-
## Get Simple SVC and PyAudioAnalysis predictions for each (with sound file name)
## Listen to "for sures" "hard to says" and "no snores"
## Idea- don't need to boil the ocean.  If you can shock and VICOUS snoring first
## you can gradually over time tune model on 'lighter snores'

## You have enough in either event. Clean up these files- write next blog post, move on to Bluetooth
# controller