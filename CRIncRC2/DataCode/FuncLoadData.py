import pandas as pd

def LoadData(DataPath, PositivePath, NegativePath, Seed):
    """
    A function to load the data from the given datafile
    Input: TotalDataPath, PositiveNamePath, NegativeNamePath, and random seed for shuffle.
    Output: TrainingData(DataFrame), TrainingLabel(List) and ValidSet(DataFrame). Please note that the index in DataFrame are not reset.

    """
    Data = pd.read_csv(DataPath)
    PositiveData = pd.read_csv(PositivePath, header=None).iloc[:,0].tolist()
    NegativeData = pd.read_csv(NegativePath, header=None).iloc[:,0].tolist()
    # Generate the Positive Index
    PositiveIndex = []
    for i in range(len(PositiveData)):
        index = Data[Data['Gene_ID'] == PositiveData[i]].index.tolist()
        PositiveIndex.extend(index)
    # Generate the negative index
    NegativeIndex = []
    for i in range(len(NegativeData)):
        index = Data[Data['Gene_ID'] == NegativeData[i]].index.tolist()
        NegativeIndex.extend(index)

    # Generate the data that has the label
    UsefulIndex = PositiveIndex+NegativeIndex
    TrainingData = Data.iloc[UsefulIndex, :]
    ValidData = Data.drop(UsefulIndex)
    ValidName = ValidData['Gene_ID']
    ValidData = ValidData.drop(['Gene_ID'], axis = 1)

    # Generate the training data label
    P_Label = [1 for i in range(len(PositiveIndex))]
    N_Label = [0 for i in range(len(NegativeIndex))]
    TrainingLabel = P_Label + N_Label

    TrainingData = TrainingData.assign(Label = TrainingLabel)
    TrainingData = TrainingData.sample(frac=1, random_state=Seed)

    TrainingLabel = TrainingData['Label'].tolist()
    TrainingName = TrainingData['Gene_ID']
    TrainingData = TrainingData.drop(['Label', 'Gene_ID'], axis=1)


    return TrainingData, TrainingLabel, ValidData#, TrainingName, ValidName