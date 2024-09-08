import torch
import torch.nn as nn
from torchvision import models

# VGG19 Feature Extractor (used for extracting style features)
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # Using VGG-19 up to relu4_2 for feature extraction
        vgg = models.vgg19(pretrained=True).features
        self.layers = {
            '0': 'relu1_1',  # First convolutional layer
            '5': 'relu2_1',  # After the second max-pooling layer
            '10': 'relu3_1',  # After the third max-pooling layer
            '19': 'relu4_1',  # After the fourth max-pooling layer
        }
        self.model = nn.Sequential(*[vgg[i] for i in range(21)])  # Up to 'relu4_1'

    def forward(self, x):
        """
        Extract features from different layers of VGG-19.
        Args:
        - x: Input image tensor
        Returns:
        - features: A dictionary of feature maps from selected layers
        """
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features  # Return features from multiple layers

# Gram matrix computation function
def gram_matrix(tensor):
    """
    Compute the Gram matrix for a given tensor.
    Args:
    - tensor: Input feature tensor of shape (B, C, H, W).
    Returns:
    - Gram matrix of the input tensor.
    """
    B, C, H, W = tensor.size()
    features = tensor.view(B, C, H * W)
    G = torch.bmm(features, features.transpose(1, 2))  # Gram matrix computation
    return G / (C * H * W)

# MyModel (style transfer model) that uses the VGG feature extractor for style features
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Define a convolutional neural network for style transfer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)

        # Downsampling layers
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Residual layers (commonly used in style transfer networks)
        self.res1 = self._residual_block(256)
        self.res2 = self._residual_block(256)
        self.res3 = self._residual_block(256)
        self.res4 = self._residual_block(256)
        self.res5 = self._residual_block(256)

        # Upsampling layers
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

        # TanH activation for the final output
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
        - x: Input tensor.
        Returns:
        - Transformed output tensor.
        """
        # Pass input through the downsampling layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Pass through the residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # Pass through the upsampling layers
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.tanh(self.conv6(x))

        # Rescale the output image back to [0, 255]
        return (x + 1) / 2 * 255

    def _residual_block(self, in_channels):
        """
        Create a residual block with two convolutional layers.
        Args:
        - in_channels: Number of input channels.
        Returns:
        - Residual block (sequential model).
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels, affine=True)
        )
        return block
