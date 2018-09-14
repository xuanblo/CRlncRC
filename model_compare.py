# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:05:02 2017

@author: ZHANGXUAN
"""

# go random forest best feature

import os
from datetime import datetime
from sklearn import metrics
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
import random
import pickle
import multiprocessing
cpu = int(multiprocessing.cpu_count() / 2 - 1)

import string
import time

KEY_LEN = 20

 
def key_gen():
    keylist = [random.choice('abcdefghijklmnopqrstuvwxyz123456789') for i in range(KEY_LEN)]
    return ("".join(keylist))


def best_feature(df):
    from sklearn.feature_selection import RFECV
    #from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    y = np.array(df.label)
    X=np.array(df.iloc[:,1:].values)
    #clf2 = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=1, n_jobs = cpu)
    forest = ExtraTreesClassifier(n_estimators=100,
                                  random_state=0, n_jobs = -1)
    rfecv = RFECV(forest, step=1, cv=10, n_jobs = -1) #n_jobs = 16
    rfecv.fit(X, y)

#    plt.close()  
    print("Optimal number of features : %d" % rfecv.n_features_)
    names = list(df.columns)
    result_list = sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), names))
    best_feature_list = []
    for index, name in result_list:
        print(index, name)
        if len(best_feature_list) < rfecv.n_features_:
            best_feature_list.append(name)
        
    #print(rfecv.ranking_, rfecv.support_)
    #print(best_feature_list)
    return rfecv.n_features_, best_feature_list

def feature_select(df):
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import ExtraTreesClassifier
    y = np.array(df.label)
    X=np.array(df.iloc[:,1:].values)
    
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=100,
                                  random_state=0, n_jobs = -1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    
    best_features = ['label']
    #save_feature = open("save_feature.txt",'w')
    for f in range(X.shape[1]):
        best_features.append(df.columns[indices[f]+1])

    n_best = len(best_features)
    df = df.loc[:,best_features[0:n_best]] # 0 = label

    return df

def cross_val_roc(classifier, clf, X, y, curve = True):
    from sklearn.model_selection import cross_val_score, KFold
    from scipy import interp
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 30)
    cv = 10
    k_fold = KFold(n_splits = cv, shuffle = True)
    k_scores = []
    recall_score = []
    precision_score = []
    accuracy_score = []
    for i, (train, test) in enumerate(k_fold.split(X, y)):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        score = clf.score(X[test], y[test])
        k_scores.append(score)
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        recall_score.append(metrics.recall_score(y[test],clf.predict(X[test])))
        precision_score.append(metrics.precision_score(y[test],clf.predict(X[test])))
        accuracy_score.append(metrics.accuracy_score(y[test],clf.predict(X[test])))

    mean_tpr /= cv
    mean_tpr[-1] = 1.0
    mean_tpr = list(map(lambda x:str(x), mean_tpr))
    mean_tpr_str = "\t".join(mean_tpr)
    print(mean_tpr_str)
    print("%s recall: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(recall_score).mean(), np.array(recall_score).std()))
    print("%s precision: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(precision_score).mean(), np.array(precision_score).std()))
    print("%s accuracy: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(accuracy_score).mean(), np.array(accuracy_score).std()))
    print("%s cross validation accuracy: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(k_scores).mean(), np.array(k_scores).std()))

        
# Random Forest Classifier
def random_forest_classifier(classifier, train_X, train_y, curve):
    global cpu
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, \
                                min_samples_split=2,\
                                random_state=1, n_jobs = -1)

    clf.fit(train_X,train_y)
    
    cross_val_roc(classifier, clf, train_X, train_y, curve)
    return clf
    

def naive_bayes_classifier(classifier, train_X,train_y, curve):
    global cpu
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB(alpha = 5)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf, train_X, train_y, curve)
    return clf

# KNN Classifier
def knn_classifier(classifier, train_X, train_y, curve):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf,train_X, train_y, curve)
    return clf

# Logistic Regression Classifier
def logistic_regression_classifier(classifier, train_X, train_y, curve):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=10, n_jobs = -1)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf, train_X, train_y, curve)
    return clf

# SVM Classifier using cross validation
def svm_cross_validation(classifier, train_X, train_y, curve):
    train_X = minmaxscaler(train_X)
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    clf = SVC(kernel = 'rbf', probability = True)
    C_range = np.logspace(-2, 10 ,13)
    gamma_range = np.logspace(-9, 3, num = 13)
    param_grid = dict(gamma = gamma_range, C = C_range)
    grid_search = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1)
    grid_search.fit(train_X,train_y)
    best_parameters = grid_search.best_params_
    for para, val in best_parameters.items():
        print(para , val)
    clf = SVC(kernel='rbf', C= best_parameters['C'], gamma = best_parameters['gamma'], probability = True)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf, train_X, train_y, curve)
    return clf

# Data Normalization
def data_normalize(X):
    from sklearn import preprocessing
    # normalize the data attributes
    normalized_X = preprocessing.normalize(X)
    # standardized_X = preprocessing.scale(X)
    standardized_X = preprocessing.scale(normalized_X)
    return standardized_X

def minmaxscaler(X_train):
    
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train=min_max_scaler.fit_transform(X_train)
    return X_train
    

# read data
def random_negetive_set(negative_lncRNA_list):
    global file_name
    df = pd.read_csv(data_file, sep = '\t', index_col = 0, header = 0)

    flag0 = 1 # use dataframe function fillna
    if flag0 != 0:
        df = df.fillna(method = 'pad')
    
    with open("negative.lncRNA.glist.xls", "r") as f:
        #
        negative_list = f.readlines()
        negative_list = list(map(lambda x:x.strip(), negative_list))
        print("negative_list:", len(set(negative_list)))
        f.close()

    random_negative_list = list(random.sample(negative_list, 150))
    
    file_random_negative = "{}.random_negative.pickle".format(file_name)
    file_random_negative_path = os.path.join("background", file_random_negative)
    print("background gene set: {}".format(file_random_negative))
    with open(file_random_negative_path, "wb") as f:
        pickle.dump(random_negative_list, f)
        f.close()  
    

def ture_positive(data_file):
    global file_name
    df = pd.read_csv(data_file, sep = '\t', index_col = 0, header = 0)

    flag0 = 1 # use dataframe function fillna
    if flag0 != 0:
        df = df.fillna(method = 'pad')
    with open("positive.lncRNA.glist.xls", "r") as f:
        positive_list = f.readlines()
        positive_list = list(map(lambda x:x.strip(), positive_list))
        print("true possitive set:", len(set(positive_list)))
        f.close()
    
        
    if True:
        file_random_negative = "{}.random_negative.pickle".format(file_name)
        file_random_negative_path = os.path.join("background", file_random_negative)
        with open(file_random_negative_path, "rb") as f:
            negative_list = pickle.load(f)
            print("negetive set:", len(set(negative_list)))
            f.close()

    positive_df = df.loc[positive_list,:]
    negative_df = df.loc[negative_list,:]
    positive_df['label'] = 1
    negative_df['label'] = -1
               
    frames = [positive_df, negative_df]
      
    df1 = pd.concat(frames)

    cols = list(df1.columns)
    new_cols = cols[:-1]
    new_cols.insert(0,'label')
    df1 = df1.loc[:,new_cols]
    
    # feature selection
    df1 = feature_select(df1)
    #df1 = feature_select2(df1)
    
    y = np.array(df1.label)

   
    X=np.array(df1.iloc[:,1:].values)
    
    return X, y


if __name__ == '__main__':
    
    random_key = key_gen()
    time_str = time.time()
    file_name = random_key + str(time_str)
    os.makedirs('background', exist_ok = True)
    data_file = 'bigtable.txt'

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'SVMCV']
    classifiers = {'NB': naive_bayes_classifier,
               'KNN': knn_classifier,
               'LR': logistic_regression_classifier,
               'RF': random_forest_classifier,
               'SVMCV': svm_cross_validation,
        }

    start_time = datetime.now()
    print("start time:", start_time)
    random_negetive_positive(data_file)

    X, y = ture_positive(data_file)

    print("=====True positive")
    for classifier in test_classifiers:
        print("\n>>> *** %s ***" % (classifier))
        clf = classifiers[classifier](classifier, X, y, curve = True)
    end_time = datetime.now()
    print("end time:", end_time)
    print('cost: {}'.format(end_time - start_time))
    
