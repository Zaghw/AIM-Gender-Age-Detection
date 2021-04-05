import pandas as pd
import numpy as np

def getClassesDataFrames(originalDF, classEdges):
    classesDFs = []
    datasetSize = 0
    for index, edge in enumerate(classEdges):
        if index != len(classEdges) - 1:  # not last class edge
            newDFFemales = originalDF[(originalDF['ages'] >= edge) & (originalDF['ages'] < classEdges[index + 1]) & (originalDF['genders'] == 0)]
            newDFMales = originalDF[(originalDF['ages'] >= edge) & (originalDF['ages'] < classEdges[index + 1]) & (originalDF['genders'] == 1)]
        else:
            newDFFemales = originalDF[(originalDF['ages'] >= edge) & (originalDF['genders'] == 0)]
            newDFMales = originalDF[(originalDF['ages'] >= edge) & (originalDF['genders'] == 1)]
        datasetSize += len(newDFFemales)
        datasetSize += len(newDFMales)
        classesDFs.append([newDFFemales, newDFMales])
    return classesDFs, datasetSize

def getClassesDistribution(classes):
    print("###################################################")

    datasetSize = 0
    for index, dfList in enumerate(classes):
        datasetSize += len(dfList[0])  # Females
        datasetSize += len(dfList[1])  # Males

    totalFemales = 0
    totalMales = 0
    for index, dfList in enumerate(classes):
        nFemales = len(dfList[0])
        nMales = len(dfList[1])
        classSize = nMales+nFemales
        totalMales += nMales
        totalFemales += nFemales

        print("Class" + str(index) + ":\t", classSize, "\t", round(100*classSize/datasetSize, 2), "\t%", end="\t")
        print("Females:\t", round(nFemales*100/classSize, 2), "\t%", end="\t")
        print("Males:\t", round(nMales*100/classSize, 2), "\t%")


    print("Total:\t", datasetSize, end="\t\t\t\t")
    print("Females:\t", round(totalFemales*100/datasetSize, 2), "\t%", end="\t")
    print("Males:\t", round(totalMales*100/datasetSize, 2), "\t%")

    print("###################################################")

def balanceGenders(classes, genderMajToMinRatio):
    balancedClasses = []

    for index, dfList in enumerate(classes):

        minGenderCount = min(len(dfList[0]), len(dfList[1]))
        maxGenderCount = int(minGenderCount*genderMajToMinRatio)
        nFemales = min(len(dfList[0]), maxGenderCount)
        nMales = min(len(dfList[1]), maxGenderCount)

        balancedFemales = dfList[0].sample(nFemales)
        balancedMales = dfList[1].sample(nMales)

        balancedClasses.append([balancedFemales, balancedMales])

    return balancedClasses


def balanceClasses(classes, classMajToMinRatio):
    balancedClasses = []

    minClassCount = np.inf
    for index, dfList in enumerate(classes):
        classCount = len(dfList[0]) + len(dfList[1])
        if classCount < minClassCount:
            minClassCount = classCount

    maxClassCount = int(minClassCount*classMajToMinRatio)

    for index, dfList in enumerate(classes):
        classCount = len(dfList[0]) + len(dfList[1])
        if classCount <= maxClassCount:
            balancedClasses.append([dfList[0], dfList[1]])
            continue
        nFemales = len(dfList[0])
        nMales = len(dfList[1])
        nGenderDiff = abs(nFemales - nMales)

        nDropTotal = classCount - maxClassCount  # Total number of samples that need to be dropped from this class
        # Get number of samples that need to be dropped from each gender
        # We prioritize dropping samples from the larger gender until they are balanced then we drop evenly
        if nFemales > nMales:
            nDropMales = max(int((nDropTotal - nGenderDiff)/2), 0)
            nDropFemales = nDropTotal - nDropMales
        else:
            nDropFemales = max(int((nDropTotal - nGenderDiff)/2), 0)
            nDropMales = nDropTotal - nDropFemales

        nFemales -= nDropFemales
        nMales -= nDropMales

        balancedFemales = dfList[0].sample(nFemales)
        balancedMales = dfList[1].sample(nMales)

        balancedClasses.append([balancedFemales, balancedMales])

    return balancedClasses


def distributeDatset(preprocessedFolderName):

    # Path variables
    DATASETS_PATH = "../Datasets/"
    PREPROCESSED_CSV_PATH = DATASETS_PATH + preprocessedFolderName + "/CSVs/"

    # Define dataset parameters
    TEST_SPLIT = 0.05  # percentage of original dataset to be used for testing
    VALID_SPLIT = 0.05  # percentage of dataset remaining after testing is removed
    classEdges = [13, 25, 35, 50]
    desiredDist = np.asarray([32.3, 31.7, 21.9, 14.1], dtype=float)  # Distribution of each class as a percentage of the dataset
    genderMajToMinRatio = 1.1  # Maximum Majority Class to Minority Class ratio
    classMajToMinRatio = 1.5  # Maximum Majority Class to Minority Class ratio

    # Read preprocessed dataset and divide into desired classes
    preprocessedDataset = pd.read_csv(PREPROCESSED_CSV_PATH + "preprocessedDataset.csv", index_col=0).reset_index(drop=True) #ensure index values are unique for each row
    classesDFs, datasetSize = getClassesDataFrames(preprocessedDataset, classEdges)

    # Prepare test, validation, and training dataset DFs
    testDataset = pd.DataFrame(columns=["genders", "ages", "img_paths"])
    validDataset = pd.DataFrame(columns=["genders", "ages", "img_paths"])
    trainDataset = pd.DataFrame(columns=["genders", "ages", "img_paths"])
    nTest = int(datasetSize * TEST_SPLIT)
    nValid = int((datasetSize - nTest) * VALID_SPLIT)
    nTrain = datasetSize - nTest - nValid
    testClassDist = (nTest * desiredDist / 100).astype(int)
    validClassDist = (nValid * desiredDist / 100).astype(int)

    # Sample from original dataset into testing and remove the samples from original dataset
    for index, nSamples in enumerate(testClassDist):
        # Sample from females and males
        testClassFemaleSamples = classesDFs[index][0].sample(int(nSamples/2))  # Female samples
        testClassMaleSamples = classesDFs[index][1].sample(int(nSamples/2))  # Male samples

        # Remove samples from original dataset
        classesDFs[index][0] = classesDFs[index][0].drop(testClassFemaleSamples.index)
        classesDFs[index][1] = classesDFs[index][1].drop(testClassMaleSamples.index)

        # Append samples to test dataset
        testDataset = testDataset.append(testClassFemaleSamples)
        testDataset = testDataset.append(testClassMaleSamples)

    # Sample from original dataset into validation and remove the samples from original dataset
    for index, nSamples in enumerate(testClassDist):
        # Sample from females and males
        validClassFemaleSamples = classesDFs[index][0].sample(int(nSamples / 2))  # Female samples
        validClassMaleSamples = classesDFs[index][1].sample(int(nSamples / 2))  # Male samples

        # Remove samples from original dataset
        classesDFs[index][0] = classesDFs[index][0].drop(validClassFemaleSamples.index)
        classesDFs[index][1] = classesDFs[index][1].drop(validClassMaleSamples.index)

        # Append samples to test dataset
        validDataset = validDataset.append(validClassFemaleSamples)
        validDataset = validDataset.append(validClassMaleSamples)

    # Balance the remaining classes to be used in training
    classesDFs = balanceClasses(balanceGenders(classesDFs, genderMajToMinRatio), classMajToMinRatio)

    # Combine remaining classes into training dataset
    for i in range(len(classesDFs)):
        trainDataset = trainDataset.append(classesDFs[i][0])
        trainDataset = trainDataset.append(classesDFs[i][1])

    testDataset.to_csv(PREPROCESSED_CSV_PATH + "test_dataset.csv")
    validDataset.to_csv(PREPROCESSED_CSV_PATH + "valid_dataset.csv")
    trainDataset.to_csv(PREPROCESSED_CSV_PATH + "train_dataset.csv")


