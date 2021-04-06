import torch
import torch.nn as nn
import torchvision.models as models
# https://discuss.pytorch.org/t/modify-resnet50-to-give-multiple-outputs/46905

class ResNet(nn.Module):
    def __init__(self, ResNetSize, num_age_classes):
        super(ResNet, self).__init__()
        if ResNetSize == "ResNet34":
            self.model_resnet = models.resnet34()
        elif ResNetSize == "ResNet50":
            self.model_resnet = models.resnet50()
        elif ResNetSize == "ResNet101":
            self.model_resnet = models.resnet101()
        elif ResNetSize == "ResNet152":
            self.model_resnet = models.resnet152()
        num_features = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        # Output for age
        self.fc1 = nn.Linear(num_features, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(num_age_classes - 1).float())

    def forward(self, x):
        x = self.model_resnet(x)

        # Age output
        age_logits = self.fc1(x)
        age_logits = age_logits + self.linear_1_bias
        age_probas = torch.sigmoid(age_logits)

        return age_logits, age_probas

def AgeOnlyResnet(ResNetSize, num_age_classes):
    if ResNetSize != "ResNet34" and ResNetSize != "ResNet50" and ResNetSize != "ResNet101" and ResNetSize != "ResNet152":
        print("Incorrect ResNet Size")
        return None

    model = ResNet(ResNetSize, num_age_classes)
    return model