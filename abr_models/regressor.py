import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
        Residual Block for ResNet1D.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the convolution. Default is 1.
            downsample (nn.Module, optional): Downsample layer. Default is None.
            dilation (int, optional): Dilation rate for the convolution. Default is 1.
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        super(ResidualBlock, self).__init__()
        padding = (5 + (dilation-1)*4) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=5, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=5, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


class ResNet1DPeakLocReg(nn.Module):
    """
        A simple 1D ResNet model for peak location regression.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden features.
            verbose (bool, optional): If True, print the shape of the tensors at each layer. Default is False.
    """

    def __init__(self, input_dim: int, hidden_dim: int, verbose: bool = False):
        super(ResNet1DPeakLocReg, self).__init__()
        self.verbose = verbose
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        self.my_log('Input:', x)
        x = self.dropout(x)
        self.my_log('After dropout:', x)
        x = self.fc(x)
        self.my_log('After fc:', x)
        x = F.relu(x)
        self.my_log('After relu:', x)
        x = self.fc2(x)
        self.my_log('After fc2:', x)
        x = F.relu(x)
        self.my_log('After relu:', x)
        x = self.fc3(x)
        self.my_log('After fc3:', x)
        return x

    def my_log(self, msg, x):
        if self.verbose:
            print(msg, x.shape)


class Regressor(nn.Module):
    """A simple 1D ResNet model for peak location regression.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden features.
        output_dim (int): Number of output features.
        verbose (bool, optional): If True, print the shape of the tensors at each layer. Default is False.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, verbose: bool = False):
        super(Regressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.verbose = verbose

        self.conv1 = nn.Conv1d(
            input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=1, dilation=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=1, dilation=4)
        self.layer4 = self._make_layer(
            256, hidden_dim, 2, stride=1, dilation=8)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.location_finder = ResNet1DPeakLocReg(hidden_dim, 256)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(
            in_channels, out_channels, stride, downsample, dilation=dilation))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(
                out_channels, out_channels, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.my_log('Input:', x)
        if self.input_dim == 1:
            base_layer = x.unsqueeze(1)
        else:
            base_layer = x

        print(f"model x:  {x.shape}")
        self.my_log('After squeeze:', x)

        x = self.conv1(base_layer)
        x = self.bn1(x)
        x = self.relu(x)
        layer0 = self.maxpool(x)

        self.my_log('After maxpool:', layer0)
        layer1 = self.layer1(layer0)
        self.my_log('After layer1:', layer1)
        layer2 = self.layer2(layer1)
        self.my_log('After layer2:', layer2)
        layer3 = self.layer3(layer2)
        self.my_log('After layer3:', layer3)
        layer4 = self.layer4(layer3)
        self.my_log('After layer4:', layer4)

        x = self.avgpool(layer4)
        self.my_log('After avgpool:', x)
        base_output = torch.flatten(x, 1)
        self.my_log('After flatten:', base_output)

        x = self.location_finder(base_output)
        return [layer0, layer1, layer2, layer3, layer4], x

    def my_log(self, msg, x):
        if self.verbose:
            print(msg, x.shape)
