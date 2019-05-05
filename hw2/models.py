import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []
        # TODO: Build the MLP architecture as described.

        active_to_class = {'relu': ReLU, 'sigmoid': Sigmoid}
        activation_class = active_to_class[activation]
        features_numbers = [in_features] + hidden_features
        for i in range(1, len(features_numbers)):
            in_num = features_numbers[i-1]
            out_num = features_numbers[i]
            blocks.append(Linear(in_num, out_num))
            blocks.append(activation_class())
            if dropout != 0:
                blocks.append(Dropout(dropout))

        blocks.append(Linear(features_numbers[-1], num_classes))
        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        filters = [in_channels] + self.filters
        for i in range(1, len(filters)):
            in_chann = filters[i-1]
            out_chann = filters[i]
            layers.append(nn.Conv2d(in_channels=in_chann, out_channels=out_chann, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if i % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2))

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        pool_amount = len(self.filters)/self.pool_every
        in_h_linear = int(in_h/(2**pool_amount))
        in_w_linear = int(in_w/(2**pool_amount))
        in_channels_linear = self.filters[-1]
        in_features_linear = in_h_linear*in_w_linear*in_channels_linear
        hidden_dimensions = [in_features_linear] + self.hidden_dims
        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        for i in range(1, len(hidden_dimensions)):
            in_size = hidden_dimensions[i-1]
            out_size = hidden_dimensions[i]
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dimensions[-1], self.out_classes))
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        filters = [in_channels] + self.filters
        for i in range(1, len(filters)):
            in_chann = filters[i-1]
            out_chann = filters[i]
            layers.append(nn.Conv2d(in_channels=in_chann, out_channels=out_chann, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if i % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2))

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        pool_amount = len(self.filters)/self.pool_every
        in_h_linear = int(in_h/(2**pool_amount))
        in_w_linear = int(in_w/(2**pool_amount))
        in_channels_linear = self.filters[-1]
        in_features_linear = in_h_linear*in_w_linear*in_channels_linear
        hidden_dimensions = [in_features_linear] + self.hidden_dims
        layers = []
        # (Linear -> ReLU)*M -> Linear
        # calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        for i in range(1, len(hidden_dimensions)):
            in_size = hidden_dimensions[i-1]
            out_size = hidden_dimensions[i]
            layers.append(nn.Linear(in_size, out_size))
           # layers.append(nn.BatchNorm2d(out_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dimensions[-1], self.out_classes))
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # Extract features from the input, run the classifier on them and
        # return class scores.
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out



