#!pip install sktime==0.13.0
#!pip install arff
#!pip install pyts

import numpy as np
import pandas as pd
import time
from scipy.io import arff
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from pyts.transformation import WEASEL
import matplotlib
import matplotlib.pyplot as plt
from progressbar import ProgressBar, AnimatedMarker, Bar
from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar, SimpleProgress
from tabulate import tabulate
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
from pyts.multivariate.transformation import WEASELMUSE
from sktime.datasets import load_from_arff_to_dataframe
import math

def train_test_prefix(X_training, X_test, Y_training, Y_test, method):

    training_time = 0
    test_time = 0

    if method == "minirocket":
    
        if(X_training.iloc[0].size > 1):  # if multivariate

            transformation = MiniRocketMultivariate() 
        
        else: # if univariate 
        
             transformation = MiniRocket()            

        transformation.fit(X_training)
        classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))

    elif method == 'weasel':
    
        if(len(X_training.shape) == 3 ): # if multivariate
            
            transformation = WEASELMUSE(word_size=2, n_bins=2, window_sizes =  [ 0.3, 0.4, 0.5,  0.6, 0.7, 0.8, 0.9], 
                                       chi2_threshold=15, sparse=False, norm_mean = False, norm_std = False)

        else:
            transformation = WEASEL(word_size=3, n_bins = 2, window_sizes =  [ 0.3, 0.4, 0.5,  0.6, 0.7, 0.8, 0.9], norm_mean = False, norm_std = False)
        
        transformation.fit(X_training, Y_training)
        classifier = LogisticRegression(max_iter = 10000)
    
    else:
        print("Unsupported method")
        return

    # -- transform training ------------------------------------------------

    time_a = time.perf_counter()
    X_training_transform = transformation.transform(X_training)
    time_b = time.perf_counter()
    training_time += time_b - time_a

    # -- transform test ----------------------------------------------------

    time_a = time.perf_counter()
    X_test_transform = transformation.transform(X_test)
    time_b = time.perf_counter()
    test_time += time_b - time_a

    # -- training ----------------------------------------------------------

    time_a = time.perf_counter()
    classifier.fit(X_training_transform, Y_training)
    time_b = time.perf_counter()
    training_time += time_b - time_a

    # -- test --------------------------------------------------------------

    time_a = time.perf_counter()
    Y_pred = classifier.predict(X_test_transform)
    time_b = time.perf_counter()
    test_time += time_b - time_a

    #print(classification_report(Y_test, Y_pred))
    
    return (Y_pred, accuracy_score(Y_test, Y_pred), f1_score(Y_test, Y_pred, average = 'weighted'), training_time, test_time)

def minirocket_strut_fav(training_data_path, test_data_path, n_splits = 5):

    X_TRAIN, Y_TRAIN = load_from_arff_to_dataframe(training_data_path)
    X_TEST, Y_TEST = load_from_arff_to_dataframe(test_data_path)
    X = X_TRAIN.append(X_TEST)
    X = X.fillna(0)
    Y = np.append(Y_TRAIN, Y_TEST)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  
    start = 9

    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros(n_splits)
    f1_scores = np.zeros(n_splits)
    best_timepoints = np.zeros(n_splits)

    fold = 0
    
    for train_index, test_index in kfold.split(X,Y):
    
        X_training, X_testing = X.iloc[train_index], X.iloc[test_index]
        Y_training, Y_testing = Y[train_index], Y[test_index]

        high = X.iloc[0,0].size
        mid = int(high/2)
        low = 9 
        while(True): 

          low_result = train_test_prefix(X_training.iloc[:,:low], X_testing.iloc[:,:low], Y_training, Y_testing, method = 'minirocket')
          mid_result = train_test_prefix(X_training.iloc[:,:mid], X_testing.iloc[:,:mid], Y_training, Y_testing, method = 'minirocket')
          high_result = train_test_prefix(X_training.iloc[:,:high], X_testing.iloc[:,:high], Y_training, Y_testing, method = 'minirocket')

          training_time[fold] += low_result[3] + mid_result[3] + high_result[3]
          test_time[fold] += low_result[4] + mid_result[4] + high_result[4]

          if math.isclose(mid_result[1], low_result[1]): 
            accuracies[fold] = low_result[1]
            f1_scores[fold] = low_result[2]
            best_timepoints[fold] = low
            break

          if math.isclose(mid_result[1], high_result[1]):
            accuracies[fold] = mid_result[1]
            f1_scores[fold] = mid_result[2]
            best_timepoints[fold] = mid
            break

          if(low_result[1] < high_result[1]):
            low = mid
            mid = int(low + (high - low)/2)
          else:
            high = mid
            mid = int(low + (high - low)/2)
          
          if((mid - low < 2 or  high - mid < 2)):
            accuracies[fold] = mid_result[1]
            f1_scores[fold] = mid_result[2]
            best_timepoints[fold] = mid
            break
        
        print(fold + 1)
        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    earliness = np.array([mid/X.iloc[0,0].size for mid in best_timepoints]).mean()
    accuracy  = accuracies.mean()
    f1 = f1_scores.mean()
    harmonic_mean = (2 * (1 - earliness) * accuracy) / ((1 - earliness) + accuracy)

    print(tabulate([["Mean Training Time", training_time],
                    ["Mean Tresting Time", test_time],
                    ["Mean Accuracy", accuracy],
                    ["Mean F1-Score", f1],
                    ["Earliness", earliness],
                    ["Harmonic Mean", harmonic_mean]
                    ], tablefmt="grid"))
    
def weasel_strut_fav(training_data_path, test_data_path, n_splits = 5):

    data = pd.DataFrame(arff.loadarff(training_data_path)[0]).append(pd.DataFrame(arff.loadarff(test_data_path)[0]))
    X = data.iloc[:,:-1]
    Y = data.iloc[: ,-1].values.ravel()
    if (X.values[0][0].size > 1):

        a = list()
        for i in range(X.values.size): #for each sample
            b = list()
            for j in range(X.values[i][0].size): #for each dim
                b.append(list(X.values[i][0][j]))
            a.append(b)

        X = np.array(a)
        #Y = data.iloc[:,-1:].values
        ts_length = X.shape[2]
    else:
        X = X.values
        #Y = Y.values
        ts_length = X.shape[1]

    X = np.nan_to_num(X).astype(np.float32)
    label_encoder = LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  
    start = 11

    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros(n_splits)
    f1_scores = np.zeros(n_splits)
    best_timepoints = np.zeros(n_splits)

    fold = 0
    
    for train_index, test_index in kfold.split(X,Y):
    
        X_training, X_testing = X[train_index], X[test_index]
        Y_training, Y_testing = Y[train_index], Y[test_index]

        high = ts_length
        mid = int(high/2)
        low = 11 
        while(True): 

          low_result = train_test_prefix(X_training[:,:low], X_testing[:,:low], Y_training, Y_testing, method = 'weasel')
          mid_result = train_test_prefix(X_training[:,:mid], X_testing[:,:mid], Y_training, Y_testing, method = 'weasel')
          high_result = train_test_prefix(X_training[:,:high], X_testing[:,:high], Y_training, Y_testing, method = 'weasel')

          training_time[fold] += low_result[3] + mid_result[3] + high_result[3]
          test_time[fold] += low_result[4] + mid_result[4] + high_result[4]

          training_time[fold] += low_result[3] + mid_result[3] + high_result[3]
          test_time[fold] += low_result[4] + mid_result[4] + high_result[4]

          if math.isclose(mid_result[1], low_result[1]): 
            accuracies[fold] = low_result[1]
            f1_scores[fold] = low_result[2]
            best_timepoints[fold] = low
            break

          if math.isclose(mid_result[1], high_result[1]):
            accuracies[fold] = mid_result[1]
            f1_scores[fold] = mid_result[2]
            best_timepoints[fold] = mid
            break

          if(low_result[1] < high_result[1]):
            low = mid
            mid = int(low + (high - low)/2)
          else:
            high = mid
            mid = int(low + (high - low)/2)
          
          if((mid - low < 2 or  high - mid < 2)):
            accuracies[fold] = mid_result[1]
            f1_scores[fold] = mid_result[2]
            best_timepoints[fold] = mid
            break

        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    earliness = np.array([mid/ts_length for mid in best_timepoints]).mean()
    accuracy  = accuracies.mean()
    f1 = f1_scores.mean()
    harmonic_mean = (2 * (1 - earliness) * accuracy) / ((1 - earliness) + accuracy)

    print(tabulate([["Mean Training Time", training_time],
                    ["Mean Tresting Time", test_time],
                    ["Mean Accuracy", accuracy],
                    ["Mean F1-Score", f1],
                    ["Earliness", earliness],
                    ["Harmonic Mean", harmonic_mean]
                    ], tablefmt="grid"))

    
