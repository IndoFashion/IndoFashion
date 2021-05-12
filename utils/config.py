""" Basic configuration and settings for training the model"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101

BASE_DIR = "**************** Absolute path for the project directory ****************"
DATA_DIR = "**************** Absolute path for the data directory ****************"
train_file = DATA_DIR + "annotations/train_data.json"
val_file = DATA_DIR + "annotations/val_data.json"
test_file = DATA_DIR + "annotations/test_data.json"
batch_size = 32
lr = 1e-3
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ce_criterion = nn.CrossEntropyLoss()
patience = 25

MODELS = {"resnet18": resnet18(pretrained=True),
          "resnet50": resnet50(pretrained=True),
          "resnet101": resnet101(pretrained=True)}
