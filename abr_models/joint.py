import torch
import torch.nn as nn


class JointModel(nn.Module):
    """
    A joint model for peak location regression and binary classification.
    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden features.
        verbose (bool, optional): If True, print the shape of the tensors at each layer. Default is False.
    """

    def __init__(self, regressor: nn.Module, classifier: nn.Module):
        super(JointModel, self).__init__()
        self.regressor = regressor
        self.classifier = classifier
        self.classifier.base_model = self.regressor
        print(self.classifier.base_model)

    def load_state_dict(self, state_dict_path: str):
        """
        Load the state dict into the model.
        Args:
            state_dict (dict): The state dict to load.
        """
        self.classifier.load_state_dict(
            torch.load(state_dict_path, weights_only=True))

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            tuple: Tuple containing the output of the classifier and the regressor.
        """
        exist, loc = self.classifier(x)
        exist = torch.sigmoid(exist) >= 0.5
        return exist, loc
