bestMargin = -1
bestValidCost = -1

for margin in range(0, 51, 5):
    validCost = 5
    if validCost < bestValidCost or bestValidCost == -1:
        bestMargin = margin
        bestValidCost = validCost

print(bestValidCost)