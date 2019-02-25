# %%
from FuncLoadData import LoadData
import numpy as np

DataFilePath = 'OriginalData/Data.csv'
PositivePath = 'OriginalData/Positive.csv'
NegativePath = 'OriginalData/Negative.csv'

ShuffleSeed = 442
X, y, Valid = LoadData(DataFilePath, PositivePath, NegativePath, ShuffleSeed)

X = X.fillna(0)
# X = np.asarray(X)

Valid = Valid.fillna(0)
# Valid = np.asarray(Valid)

FeatureName = X.columns.values
# print (FeatureName)

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold = 1)
selector.fit(X)
RemainedFeatureLoc = selector.get_support(indices=True)
# np.savetxt("VarianceFilteredLoc.txt", RemainedFeatureLoc, fmt='%d')
X = selector.transform(X)
Valid = selector.transform(Valid)

# %%
RemainedFeatureName = FeatureName[RemainedFeatureLoc]
import pandas as pd

TrainingSet = pd.DataFrame(X, index=None, columns=RemainedFeatureName)
TrainingSet.to_csv('ProcessedData/TrainingSet.csv', index=None)

ValidSet = pd.DataFrame(Valid, index=None, columns=RemainedFeatureName)
ValidSet.to_csv('ProcessedData/ValidSet.csv', index=None)

np.savetxt('ProcessedData/TrainingLabel.txt', y, fmt='%d')

