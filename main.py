from preprocessDataset import preprocessDataset
from trainModel import trainModel
from distributeDataset import distributeDatset
from testModel import testModel

if __name__ == "__main__":
    Margin = 25
    ResNetSize = "ResNet50"
    preprocessedFolderName = "Preprocessed-" + str(Margin)
    outputFolderName = ResNetSize + "Margin" + str(Margin)

    # preprocessDataset(Margin, preprocessedFolderName)
    # distributeDatset(preprocessedFolderName)
    validCost = trainModel(ResNetSize, preprocessedFolderName, outputFolderName)
    print("Margin: ", Margin, " returned Validation Cost: ", validCost)
    testModel(ResNetSize, preprocessedFolderName, outputFolderName)

    Margin = 40
    ResNetSize = "ResNet50"
    preprocessedFolderName = "Preprocessed-" + str(Margin)
    outputFolderName = ResNetSize + "Margin" + str(Margin)

    preprocessDataset(Margin, preprocessedFolderName)
    distributeDatset(preprocessedFolderName)
    validCost = trainModel(ResNetSize, preprocessedFolderName, outputFolderName)
    print("Margin: ", Margin, " returned Validation Cost: ", validCost)
    testModel(ResNetSize, preprocessedFolderName, outputFolderName)
