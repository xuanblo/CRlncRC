#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:07:40 2018

@author: zhangxuan
"""

# calculate evaluation indicators

import pandas as pd
import numpy as np





# all features
RF_recall_score = pd.read_csv('RF.recall.list', header = None, index_col = 0).index
RF_precision_score = pd.read_csv('RF.precision.list', header = None, index_col = 0).index
RF_accuracy_score = pd.read_csv('RF.accuracy.list', header = None, index_col = 0).index

print("%s recall: %0.2f (+/- %0.2f)" % ('RF features', \
      np.array(RF_recall_score).mean(), np.array(RF_recall_score).std()))
print("%s precision: %0.2f (+/- %0.2f)" % ('RF features', \
      np.array(RF_precision_score).mean(), np.array(RF_precision_score).std()))
print("%s accuracy: %0.2f (+/- %0.2f)" % ('RF features', \
      np.array(RF_accuracy_score).mean(), np.array(RF_accuracy_score).std()))

#network
NB_recall_score = pd.read_csv('NB.recall.list', header = None, index_col = 0).index
NB_precision_score = pd.read_csv('NB.precision.list', header = None, index_col = 0).index
NB_accuracy_score = pd.read_csv('NB.accuracy.list', header = None, index_col = 0).index

print("%s recall: %0.2f (+/- %0.2f)" % ('NB_accuracy_score network features', \
      np.array(NB_recall_score).mean(), np.array(NB_recall_score).std()))
print("%s precision: %0.2f (+/- %0.2f)" % ('NB_accuracy_score network features', \
      np.array(NB_precision_score).mean(), np.array(NB_precision_score).std()))
print("%s accuracy: %0.2f (+/- %0.2f)" % ('NB_accuracy_score network features', \
      np.array(NB_accuracy_score).mean(), np.array(NB_accuracy_score).std()))

#genomic
LR_recall_score = pd.read_csv('LR.recall.list', header = None, index_col = 0).index
LR_precision_score = pd.read_csv('LR.precision.list', header = None, index_col = 0).index
LR_accuracy_score = pd.read_csv('LR.accuracy.list', header = None, index_col = 0).index

print("%s recall: %0.2f (+/- %0.2f)" % ('LR genomic features', \
      np.array(LR_recall_score).mean(), np.array(LR_recall_score).std()))
print("%s precision: %0.2f (+/- %0.2f)" % ('LR genomic features', \
      np.array(LR_precision_score).mean(), np.array(LR_precision_score).std()))
print("%s accuracy: %0.2f (+/- %0.2f)" % ('LR genomic features', \
      np.array(LR_accuracy_score).mean(), np.array(LR_accuracy_score).std()))

#expression
KNN_recall_score = pd.read_csv('KNN.recall.list', header = None, index_col = 0).index
KNN_precision_score = pd.read_csv('KNN.precision.list', header = None, index_col = 0).index
KNN_accuracy_score = pd.read_csv('KNN.accuracy.list', header = None, index_col = 0).index

print("%s recall: %0.2f (+/- %0.2f)" % ('KNN expression features', \
      np.array(KNN_recall_score).mean(), np.array(KNN_recall_score).std()))
print("%s precision: %0.2f (+/- %0.2f)" % ('KNN expression features', \
      np.array(KNN_precision_score).mean(), np.array(KNN_precision_score).std()))
print("%s accuracy: %0.2f (+/- %0.2f)" % ('KNN expression features', \
      np.array(KNN_accuracy_score).mean(), np.array(KNN_accuracy_score).std()))

#epigenetic
SVMCV_recall_score = pd.read_csv('SVMCV.recall.list', header = None, index_col = 0).index
SVMCV_precision_score = pd.read_csv('SVMCV.precision.list', header = None, index_col = 0).index
SVMCV_accuracy_score = pd.read_csv('SVMCV.accuracy.list', header = None, index_col = 0).index

print("%s recall: %0.2f (+/- %0.2f)" % ('SVMCV epigenetic features', \
      np.array(SVMCV_recall_score).mean(), np.array(SVMCV_recall_score).std()))
print("%s precision: %0.2f (+/- %0.2f)" % ('SVMCV epigenetic features', \
      np.array(SVMCV_precision_score).mean(), np.array(SVMCV_precision_score).std()))
print("%s accuracy: %0.2f (+/- %0.2f)" % ('SVMCV epigenetic features', \
      np.array(SVMCV_accuracy_score).mean(), np.array(SVMCV_accuracy_score).std()))