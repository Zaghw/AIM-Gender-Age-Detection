import torch
import torch.nn as nn

# x = torch.tensor([1.0, -1.0])
# y = torch.tensor([5, 6])
# print(torch.sum((x == y)))
# in_features = x.shape[1]  # = 2
# out_features = 1
#
# y = nn.Parameter(torch.zeros(5).float())
#
# m = nn.Linear(in_features, out_features, bias=False)
#
# test = 0
#
# test += (5 == 5)
# print(test)
hello = [{"hi": 0},{},{}]
hello[0]["hi"] += 1
hello[0]["hi"] += 1
hello[0]["hi"] += 1

print(hello[0]["hi"])