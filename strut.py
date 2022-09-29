!pip install sktime
!pip install arff
!pip install pyts

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

def train_test_prefix(X_training, X_test, Y_training, Y_test, method):

  training_time = 0
    test_time = 0

    if method == "minirocket":
    
        if(X_training.iloc[0].size > 1):  # if multivariate

            transformation = MiniRocketMultivariate( ) 
        
        else: # if univariate 
        
             transformation = MiniRocket( )            

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

def minirocket_strut(training_data_path, test_data_path, n_splits = 5, dataset = None):

    X_TRAIN, Y_TRAIN = load_from_arff_to_dataframe(training_data_path)
    X_TEST, Y_TEST = load_from_arff_to_dataframe(test_data_path)
    X = X_TRAIN.append(X_TEST)
    X = X.fillna(0) # replace NaN with 0 
    Y = np.append(Y_TRAIN, Y_TEST)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  

    start = 9
    ts_length = X.iloc[0][0].size

    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros((n_splits,ts_length+1-start))
    f1_scores = np.zeros((n_splits,ts_length+1-start))

    fold = 0
    
    for train_index, test_index in kfold.split(X,Y):
    
        X_training, X_testing = X.iloc[train_index], X.iloc[test_index]
        Y_training, Y_testing = Y[train_index], Y[test_index]

        pbar = ProgressBar(widgets=['Fold: ' + str(fold+1)+ " | ", SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
               ' ', AdaptiveETA()])
        for i in pbar(range(start,ts_length+1)): 
  
            result = train_test_prefix(X_training.iloc[:,:i], X_testing.iloc[:,:i], Y_training, Y_testing, method = "minirocket")
            #returns (predictions, accuracy, f1 score, training time, test time)

            accuracies[fold][i-start] = result[1]
            f1_scores[fold][i-start] =  result[2]
            training_time[fold] += result[3]
            test_time[fold] += result[4]

        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    accuracies_mean = accuracies.mean(axis=0)
    accuracies_std = accuracies.std()
    accuracies_ci = 0.1 * accuracies_std / accuracies_mean

    f1_scores_mean = f1_scores.mean(axis=0)
    f1_scores_std = f1_scores.std()
    f1_scores_ci = 0.1 * f1_scores_std / f1_scores_mean

    earlinesses = np.array([float(x)/ts_length for x in range(start, ts_length+1)])

    harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)

    best_accuracy = accuracies_mean.max()
    best_accuracy_timepoint = np.argmax(accuracies_mean)

    best_f1_score = f1_scores_mean.max()
    best_f1_score_timepoint = np.argmax(f1_scores_mean)

    best_harmonic_mean = harmonic_means.max()
    best_harmonic_mean_timepoint = np.argmax(harmonic_means)

    print(tabulate([["Mean Training Time", training_time]], tablefmt="grid"))
    
    print(tabulate([['Accuracy', best_accuracy, str(best_accuracy_timepoint+start) + '/' + str(ts_length), earlinesses[best_accuracy_timepoint], harmonic_means[best_accuracy_timepoint]], 
                    ['F1-score', best_accuracy, str(best_f1_score_timepoint+start) + '/' + str(ts_length), earlinesses[best_f1_score_timepoint], harmonic_means[best_f1_score_timepoint]],
                    ['Harmonic Mean', best_harmonic_mean, str(best_harmonic_mean_timepoint+start) + '/' + str(ts_length), earlinesses[best_harmonic_mean_timepoint], harmonic_means[best_harmonic_mean_timepoint]]                    
                   ], headers=['Metric', 'Best Value', 'Timepoint', 'Earliness', 'Harmonic Mean'], tablefmt="grid"))

    timepoints = np.arange(start,ts_length+1) 

    #plot accuracy
    full_time_accuracy = np.repeat(accuracies_mean[-1], timepoints.shape[0] )
 
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints, full_time_accuracy , '--', color = 'blue', label = "Minirocket Full-Time")
    ax.plot(timepoints, accuracies_mean, color = 'blue', label = "Minirocket STRUT") 
    ax.fill_between(timepoints,(accuracies_mean - accuracies_ci), (accuracies_mean + accuracies_ci), color='blue', alpha=0.4)
    ax.plot([best_accuracy_timepoint+start], [best_accuracy], 'D', markersize = 10,  label = 'Best Accuracy')
    plt.xlabel("Truncation Timepoint")
    plt.ylabel("Accuracy")
    plt.ylim(0.0,1.1)
    plt.legend() 
    plt.show()

    #plot f1 score
    full_time_f1 = np.repeat(f1_scores_mean[-1], timepoints.shape[0] )
 
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints, full_time_f1 , '--', color = 'blue', label = "Minirocket Full-Time")
    ax.plot(timepoints, f1_scores_mean, color = 'blue', label = "Minirocket STRUT" ) 
    ax.fill_between(timepoints,(f1_scores_mean - f1_scores_ci), (f1_scores_mean + f1_scores_ci), color='blue', alpha=0.4)
    ax.plot([best_f1_score_timepoint+start], [best_f1_score], 'D', markersize = 10, label = 'Best F1-score')
    plt.xlabel("Truncation Time-point")
    plt.ylabel("F1-score")
    plt.ylim(0.0,1.1)
    plt.legend() 
    plt.show()

    #plot harmonic mean
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints , harmonic_means , color='blue', label = "Minirocket STRUT" ) 
    ax.plot([best_harmonic_mean_timepoint+start], [best_harmonic_mean], 'D', markersize = 10,  label = 'Best Harmonic Mean')
    plt.ylabel("Harmonic Mean")
    plt.xlabel("Truncation Time-point")
    plt.ylim(0.0,1.1)
    plt.legend() 
    plt.show()

    res = (accuracies_mean, accuracies_std, accuracies_ci, (best_accuracy_timepoint+start, best_accuracy),
            f1_scores_mean, f1_scores_std, f1_scores_ci, (best_f1_score_timepoint+start, best_f1_score),
            earlinesses, harmonic_means,
            training_time, test_time)
       
    if dataset != None:
      np.save(dataset + " _minirocket_results.npy", np.array(res, dtype=object))

    return res

def weasel_strut(training_data_path, test_data_path, n_splits = 5, dataset = None):

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

    X = np.nan_to_num(X).astype(np.float32) # replace NaN with 0 
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  

    start = 11
    
    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros((n_splits,ts_length+1-start))
    f1_scores = np.zeros((n_splits,ts_length+1-start))

    fold = 0
    
    for train_index, test_index in kfold.split(X,Y):
    
        X_training, X_testing = X[train_index], X[test_index]
        Y_training, Y_testing = Y[train_index], Y[test_index]

        pbar = ProgressBar(widgets=['Fold: ' + str(fold+1)+ " | ", SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
               ' ', AdaptiveETA()])
        for i in pbar(range(start,ts_length+1)): 
            
            result = train_test_prefix(X_training[:,:i], X_testing[:,:i], Y_training, Y_testing, method = 'weasel')
            #returns (predictions, accuracy, f1 score, training time, test time)

            accuracies[fold][i-start] = result[1]
            f1_scores[fold][i-start] =  result[2]
            training_time[fold] += result[3]
            test_time[fold] += result[4]

        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    accuracies_mean = accuracies.mean(axis=0)
    accuracies_std = accuracies.std()
    accuracies_ci = 0.1 * accuracies_std / accuracies_mean

    f1_scores_mean = f1_scores.mean(axis=0)
    f1_scores_std = f1_scores.std()
    f1_scores_ci = 0.1 * f1_scores_std / f1_scores_mean

    earlinesses = np.array([float(x)/ts_length for x in range(start, ts_length+1)])

    harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)

    best_accuracy = accuracies_mean.max()
    best_accuracy_timepoint = np.argmax(accuracies_mean)

    best_f1_score = f1_scores_mean.max()
    best_f1_score_timepoint = np.argmax(f1_scores_mean)

    best_harmonic_mean = harmonic_means.max()
    best_harmonic_mean_timepoint = np.argmax(harmonic_means)

    print(tabulate([["Mean Training Time", training_time]], tablefmt="grid"))
    
    print(tabulate([['Accuracy', best_accuracy, str(best_accuracy_timepoint+start) + '/' + str(ts_length), earlinesses[best_accuracy_timepoint], harmonic_means[best_accuracy_timepoint]], 
                    ['F1-score', best_accuracy, str(best_f1_score_timepoint+start) + '/' + str(ts_length), earlinesses[best_f1_score_timepoint], harmonic_means[best_f1_score_timepoint]],
                    ['Harmonic Mean', best_harmonic_mean, str(best_harmonic_mean_timepoint+start) + '/' + str(ts_length), earlinesses[best_harmonic_mean_timepoint], harmonic_means[best_harmonic_mean_timepoint]]                    
                   ], headers=['Metric', 'Best Value', 'Timepoint', 'Earliness', 'Harmonic Mean'], tablefmt="grid"))

    timepoints = np.arange(start,ts_length+1) 

    #plot accuracy
    full_time_accuracy = np.repeat(accuracies_mean[-1], timepoints.shape[0] )
 
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints, full_time_accuracy , '--', color = 'blue', label = "Weasel Full-Time")
    ax.plot(timepoints, accuracies_mean, color = 'blue', label = "Weasel STRUT") 
    ax.fill_between(timepoints,(accuracies_mean - accuracies_ci), (accuracies_mean + accuracies_ci), color='blue', alpha=0.4)
    ax.plot([best_accuracy_timepoint+start], [best_accuracy], 'D', markersize = 10,  label = 'Best Accuracy')
    plt.xlabel("Truncation Timepoint")
    plt.ylabel("Accuracy")
    plt.ylim(0.0,1.1)
    plt.legend() 
    plt.show()

    #plot f1 score
    full_time_f1 = np.repeat(f1_scores_mean[-1], timepoints.shape[0] )
 
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints, full_time_f1 , '--', color = 'blue', label = "Weasel Full-Time")
    ax.plot(timepoints, f1_scores_mean, color = 'blue', label = "Weasel STRUT" ) 
    ax.fill_between(timepoints,(f1_scores_mean - f1_scores_ci), (f1_scores_mean + f1_scores_ci), color='blue', alpha=0.4)
    ax.plot([best_f1_score_timepoint+start], [best_f1_score], 'D', markersize = 10, label = 'Best F1-score')
    plt.xlabel("Truncation Time-point")
    plt.ylabel("F1-score")
    plt.ylim(0.0,1.1)
    plt.legend() 
    plt.show()

    #plot harmonic mean
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints , harmonic_means , color='blue', label = "Weasel STRUT" ) 
    ax.plot([best_harmonic_mean_timepoint+start], [best_harmonic_mean], 'D', markersize = 10,  label = 'Best Harmonic Mean')
    plt.ylabel("Harmonic Mean")
    plt.xlabel("Truncation Time-point")
    plt.ylim(0.0,1.1)
    plt.legend() 
    plt.show()

    res = (accuracies_mean, accuracies_std, accuracies_ci, (best_accuracy_timepoint+start, best_accuracy),
            f1_scores_mean, f1_scores_std, f1_scores_ci, (best_f1_score_timepoint+start, best_f1_score),
            earlinesses, harmonic_means,
            training_time, test_time)
       
    if dataset != None:
      np.save(dataset + " _weasel_results.npy", np.array(res, dtype=object))

    return res
