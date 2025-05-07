import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCnnModel(nn.Module):
    """
    A simple CNN model for binary classification.
    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden features.
        output_dim (int): Number of output features.
        verbose (bool, optional): If True, print the shape of the tensors at each layer. Default is False.
    """

    def __init__(self, base_model):
        super(SimpleCnnModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv1d(64, 256, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2496, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        if self.base_model:
            layers, loc = self.base_model(x)
            first_layer = self.conv1(layers[0])
        else:
            layers = x
            loc = None
            first_layer = self.conv1(x)
        x = F.relu(self.batchnorm1(first_layer))
        x = F.avg_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = torch.cat((x, layers[1]), 2)
        x = F.avg_pool1d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        exist = self.fc4(x)

        return exist, loc
