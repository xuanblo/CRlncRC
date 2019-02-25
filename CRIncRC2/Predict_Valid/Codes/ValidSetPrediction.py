import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler

clf = XGBClassifier(
            objective='binary:logistic', 
            colsample_bytree = 0.94, 
            gamma = 0.03, 
            learning_rate = 0.124, 
            max_depth = 10, 
            n_estimators = 1998, 
            subsample=0.718
        )


X = pd.read_csv("../../BMCProcessedData/TrainingSet.csv")
X = np.asarray(X)
y = np.loadtxt("../../BMCProcessedData/TrainingLabel.txt", dtype = int)


# Standardizing
X = StandardScaler().fit_transform(X)

# Over-sampling
from imblearn.over_sampling import SMOTE
resampler = SMOTE(kind='svm', random_state=442)
X_res, y_res = resampler.fit_resample(X, y)



clf.fit(X_res, y_res)


X_test = np.asarray(pd.read_csv('../../BMCProcessedData/ValidSet.csv'))

from sklearn.preprocessing import StandardScaler
X_test = StandardScaler().fit_transform(X_test)

y_pred = clf.predict(X_test)

print (np.where(y_pred==1)[0].shape)

np.savetxt("pred.txt", y_pred, fmt='%d')
y_prob = clf.predict_proba(X_test)[:, 1]
np.savetxt("prob.txt", y_prob, fmt='%.4f')