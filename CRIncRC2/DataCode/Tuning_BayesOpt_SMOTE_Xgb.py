# %%
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

# Prepare xgb for bayes opt
def xgb_cv(n_estimator, max_depth, learning_rate, col_bytree, gamma, subsample, data, targets):
    estimator = XGBClassifier(n_estimators=n_estimator, 
                    max_depth=max_depth, 
                    learning_rate=learning_rate, 
                    colsample_bytree=col_bytree, 
                    gamma=gamma, 
                    subsample=subsample)
    cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=10)
    return cval.mean()

def optimize_xgb(data, targets):
    """Apply Bayesian Optimization to Xgb parameters."""
    def xgb_crossval(n_estimators, max_depth, learning_rate, colsample_bytree, gamma, subsample):
        return xgb_cv(n_estimator = int(n_estimators), 
                    max_depth = int(max_depth), 
                    learning_rate = learning_rate, 
                    col_bytree = colsample_bytree,
                    gamma = gamma, 
                    subsample = subsample,
                    data=data, 
                    targets=targets)

    optimizer = BayesianOptimization(
        f=xgb_crossval,
        pbounds={'n_estimators': (10, 2000),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'colsample_bytree': (0.7, 1),
            'gamma': (0, 0.05),
            'subsample': (0.7, 1)},
        random_state=442)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        optimizer.maximize(n_iter=20, acq='ei')

    return optimizer

if __name__ == "__main__":
    # Load Data
    X = pd.read_csv("ProcessedData/TrainingSet.csv")
    y = np.loadtxt("ProcessedData/TrainingLabel.txt", dtype = int)

    # Over-sampling
    from imblearn.over_sampling import SMOTE
    resampler = SMOTE(kind='svm', random_state=442)
    X_res, y_res = resampler.fit_resample(X, y)

    # Standardizing
    X_res = StandardScaler().fit_transform(X_res)

    OptRes = optimize_xgb(X_res, y_res)
    print("Final result:", OptRes.max)

    history_df = pd.DataFrame(OptRes.res)
    history_df.to_csv('Porto-AUC-10fold-XGB.csv')


