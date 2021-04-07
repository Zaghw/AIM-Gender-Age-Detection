from preprocessDataset import preprocessDataset
# from trainModel import trainModel
from distributeDataset import distributeDatset
from trainAgeOnlyModel import trainAgeOnlyModel
from testModel import testModel

if __name__ == "__main__":
    Margin = 0
    ResNetSize = "ResNet50"
    preprocessedFolderName = "Preprocessed-" + str(Margin)
    outputFolderName = "AgeOnly" + ResNetSize + "Margin" + str(Margin)

    # preprocessDataset(Margin, preprocessedFolderName)
    # distributeDatset(preprocessedFolderName)
    validCost = trainAgeOnlyModel(ResNetSize, preprocessedFolderName, outputFolderName)
    print("Margin: ", Margin, " returned Validation Cost: ", validCost)
    testModel(ResNetSize, preprocessedFolderName, outputFolderName)


