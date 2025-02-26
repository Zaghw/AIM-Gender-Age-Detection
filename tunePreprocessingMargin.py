from preprocessDataset import preprocessDataset
from trainModel import trainModel

if __name__ == "__main__":
    bestMargin = -1
    bestValidCost = -1

    for margin in range(0, 51, 5):
        print("Margin: ", margin)
        preprocessDataset(margin)
        validCost = trainModel()
        if validCost < bestValidCost or bestValidCost == -1:
            bestMargin = margin
            bestValidCost = validCost
    print("Best Margin: ", str(bestMargin))
    print("Best Validation Cost: ", str(bestValidCost))
