"""
    Class file that enlists models for image classification
"""
import torch
import torch.nn as nn
from utils.config import MODELS


class ResNetFeaturesFlatten(nn.Module):
    def __init__(self, model_key):
        """
           Initialize the model architecture

           Args:
               model_key (str): Key value corresponding to the model architecture

           Returns:
               None
           """
        super(ResNetFeaturesFlatten, self).__init__()
        self.resnet_model = MODELS[model_key]
        self.conv1 = self.resnet_model.conv1
        self.bn1 = self.resnet_model.bn1
        self.relu = self.resnet_model.relu
        self.maxpool = self.resnet_model.maxpool
        self.layer1 = self.resnet_model.layer1
        self.layer2 = self.resnet_model.layer2
        self.layer3 = self.resnet_model.layer3
        self.layer4 = self.resnet_model.layer4
        self.avgpool = self.resnet_model.avgpool
        if model_key == "resnet18":
            self.fc = nn.Linear(512, 15)
        else:
            self.fc = nn.Linear(2048, 15)

    def forward(self, input):
        """
           Function to compute forward pass of the network

           Args:
               input (Tensor): Image tensor of shape (N X C X H X W), where N denotes minibatch size, C, H, W denotes image channels, width and height

           Returns:
               output (Tensor): Raw prediction scores for each class
           """
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.fc(self.relu(x))
        return output
