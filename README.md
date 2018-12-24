# CRlncRC (Cancer Related lncRNA Classifier)

Long noncoding RNAs (lncRNAs) are widely involved in the initiation and development of cancer. Although some computational methods have been proposed to identify cancer-related lncRNAs, there is still a demanding to improve the prediction accuracy and efficiency. In addition, the quick-update data of cancer, as well as the discovery of new mechanism, also underlay the possibility of improvement of cancer-related lncRNA prediction algorithm. In this study, we introduced CRlncRC, a novel Cancer-Related lncRNA Classifier by integrating manifold features with five machine-learning techniques.

## Features

85 features in 4 categories (genomic, epigenetic, expression, network).

## Best model

CRlncRC was built on the integration of genomic, expression, epigenetic and network, totally in four categories of features. Five learning techniques were exploited to develop the effective classification model including Random Forest (RF), Naïve bayes (NB), Support Vector Machine (SVM), Logistic Regression (LR) and K-Nearest Neighbors (KNN). Using ten-fold cross-validation, we showed that RF is the best model for classifying cancer-related lncRNAs (AUC=0.82).

## Predict cancer-related lncRNA candidates

We further applied CRlncRC to lncRNAs from the TANRIC (The Atlas of non-coding RNA in Cancer) dataset, and identified 121 cancer-related lncRNA candidates.
