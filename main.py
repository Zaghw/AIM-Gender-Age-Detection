from preprocessDataset import preprocessDataset
from trainModel import trainModel

if __name__ == "__main__":
    preprocessDataset(25)
    # validCost = trainModel("ResNet50Margin25")
    # print("Margin: ", 25, " returned Validation Cost: ", validCost)
