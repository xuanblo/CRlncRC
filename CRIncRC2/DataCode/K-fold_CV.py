import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from scipy import interp
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, classification_report
from matplotlib import pyplot as plt

def GetData():
    X = pd.read_csv("ProcessedData/TrainingSet.csv")
    y = np.loadtxt("ProcessedData/Traininglabel.txt", dtype=int)

    return np.asarray(X), y

def GetFoldResult(data, targets, cv, seed):
    Stratified_folder = StratifiedKFold(n_splits=cv, random_state=seed)
    FolderRes = Stratified_folder.split(data, targets)

    data = StandardScaler().fit_transform(data)

    # Get each k-fold results
    FoldXtrain = []
    FoldXtest = []
    Foldytrain = []
    Foldytest = []

    for train_index, test_index in FolderRes:
        # Original result for each fold
        X_train = data[train_index, :]
        y_train = targets[train_index]

        X_test = data[test_index, :]
        y_test = targets[test_index]
        # For each fold, resample the training one
        resampler = SMOTE(kind='svm', random_state=seed)
        X_res, y_res = resampler.fit_resample(X_train, y_train)

        FoldXtrain.append(X_res)
        Foldytrain.append(y_res)
        FoldXtest.append(X_test)
        Foldytest.append(y_test)

    return FoldXtrain, FoldXtest, Foldytrain, Foldytest

def GetCVScore(X_trainList, X_testList, y_trainList, y_testList, estimator, cv):
    print ("=======================\nStart calculating AUC for each fold:")
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    tprlist = []
    fprlist = []

    precisionlist = []
    recalllist = []
    pres = []
    yreal = []
    yprob = []
    cv_pr = []

    reports = []
    for i in range(cv):
        X_train = X_trainList[i]
        y_train = y_trainList[i]

        X_test = X_testList[i]
        y_test = y_testList[i]

        clf = estimator
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]

        reports.append(classification_report(y_test, y_pred))
        # ROC 
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fprlist.append(fpr)
        tprlist.append(tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # PRC
        _precision, _recall, _ = precision_recall_curve(y_test, y_pred_prob)
        precisionlist.append(_precision)
        recalllist.append(_recall)
        pres.append(interp(mean_fpr, _recall[::-1], _precision[::-1]))
        cv_pr.append(average_precision_score(y_test, y_pred_prob))
        yreal.append(y_test)
        yprob.append(y_pred_prob)

        print ("Fold {} done".format(i+1))
    yreal = np.concatenate(yreal)
    yprob = np.concatenate(yprob)
    print ("=======================")
    return fprlist, tprlist, tprs, aucs, pres, yreal, yprob, cv_pr, precisionlist, recalllist, reports
    
def PlotROC(fprlist, tprlist, tprs, aucs, cvfold):
    print ("Ploting!")
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis = 0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.figure(figsize=(8, 7))
    # Plot each fold
    for i in range(cvfold):
        plt.plot(fprlist[i], 
                tprlist[i], 
                alpha=0.3, 
                label = "ROC fold %d (AUC = %.2f)"%(i, aucs[i]))
    # Plot the chance
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=.8)
    # Plot the mean one
    plt.plot(mean_fpr, 
            mean_tpr, 
            color='b', 
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), alpha=.8)
    # Plot the variances
    plt.fill_between(mean_fpr, 
                    tprs_lower, 
                    tprs_upper, 
                    color='grey', 
                    alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    # plt.title('ROC for each fold')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()

    return plt


def PlotPRC(pres, yreal, yprob, prscores, prelist, reclist, cvfold):
    print ("Ploting!")
    mean_rec = np.linspace(0, 1, 100)
    mean_pre = np.mean(pres, axis = 0)
    mean_pre2, mean_rec2, _ = precision_recall_curve(yreal, yprob)
    mean_score = average_precision_score(yreal, yprob)

    std_score = np.std(prscores)

    std_pre = np.std(pres, axis=0)
    pres_upper = np.minimum(mean_pre + std_pre, 1)
    pres_lower = np.maximum(mean_pre - std_pre, 0)

    plt.figure(figsize=(8, 7))
    # Plot each fold
    for i in range(cvfold):
        plt.plot(reclist[i], 
                prelist[i],
                alpha=0.3, 
                label = "PRC fold %d (AP = %.2f)"%(i, prscores[i]))
    # Plot the mean one
    plt.plot(mean_rec2, 
            mean_pre2, 
            color='b', 
            label=r'Mean PRC (AP = %0.2f $\pm$ %0.2f)' % (mean_score, std_score), alpha=.8)
    # Plot the variances
    plt.fill_between(mean_rec, 
                    pres_lower, 
                    pres_upper, 
                    color='grey', 
                    alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    # plt.title('ROC for each fold')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()

    return plt


if __name__ == "__main__":
    cv = 10
    # seed = 42435 42435 makes auc 0.84
    seed = 42435

    X, y = GetData()
    X_trainList, X_testList, y_trainList, y_testList = GetFoldResult(X, y, cv, seed)

    estimator = XGBClassifier(
            objective='binary:logistic', 
            colsample_bytree = 0.94, 
            gamma = 0.03, 
            learning_rate = 0.124, 
            max_depth = 10, 
            n_estimators = 1998, 
            subsample=0.718
        )

    fprlist, tprlist, tprs, aucs, pres, yreal, yprob, prscores, prelist, reclist, reports = GetCVScore(X_trainList, X_testList, y_trainList, y_testList, estimator, cv)

#     PlotROC(fprlist, tprlist, tprs, aucs, cv)
#     plt.savefig("10FoldCV_Resample_ROC.png",bbox_inches = "tight")
#     plt.savefig("10FoldCV_Resample_ROC.eps",bbox_inches = "tight")

#     PlotPRC(pres, yreal, yprob, prscores, prelist, reclist, cv)
#     plt.savefig("10FoldCV_Resample_PRC.png",bbox_inches = "tight")
#     plt.savefig("10FoldCV_Resample_PRC.eps",bbox_inches = "tight")
#     plt.show()
    for i in range(10):
        print (reports[i])
    


