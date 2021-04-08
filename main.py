from preprocessDataset import preprocessDataset
# from trainModel import trainModel
from distributeDataset import distributeDatset
from trainAgeOnlyModel import trainAgeOnlyModel
from testModel import testModel
from testAgeOnlyModel import testAgeOnlyModel

if __name__ == "__main__":
    Margin = 0
    ResNetSize = "ResNet34"  # "ResNet34" or "ResNet50" or "ResNet101" or "ResNet152"
    preprocessedFolderName = "Preprocessed-" + str(Margin)
    outputFolderName = "AgeOnly" + ResNetSize + "Margin" + str(Margin)
    MIN_AGE = 13  # Inclusive
    MAX_AGE = 117  # Exclusive
    AGE_SEGMENTS_EDGES = [25, 35, 50]  # (MIN_AGE,24), (24,34), (35,49), (50,MAX_AGE)

    # preprocessDataset(Margin, preprocessedFolderName)
    # distributeDatset(preprocessedFolderName, MIN_AGE, MAX_AGE, AGE_SEGMENTS_EDGES)
    # validCost = trainAgeOnlyModel(ResNetSize, preprocessedFolderName, outputFolderName, MIN_AGE, MAX_AGE)
    # print("Margin: ", Margin, " returned Validation Cost: ", validCost)
    testAgeOnlyModel(ResNetSize, preprocessedFolderName, outputFolderName, MIN_AGE, MAX_AGE, AGE_SEGMENTS_EDGES)


