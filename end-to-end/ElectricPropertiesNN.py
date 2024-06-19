import numpy as np
import torch
import torch.nn as nn


class ElectricPropertiesNN(nn.Module):
    def __init__(
        self,
        kernel_size=7,
        in_channels=2,
    ):
        super(ElectricPropertiesNN, self).__init__()
        self.in_channels = in_channels
        self.average_pooling_kernel_size = kernel_size
        self.average_pooling_kernel = nn.Parameter(
            torch.ones([in_channels, 1, kernel_size, kernel_size])
            / torch.ones([in_channels, 1, kernel_size, kernel_size]).sum(),
            requires_grad=False,
        )

    def forward(self, electric_image, distances):

        processed = nn.functional.conv2d(electric_image, self.average_pooling_kernel, groups=self.in_channels)

        features = []
        for i in range(self.in_channels):
            feat = processed[:, i].reshape(self.in_channels, -1)
            feat = feat.abs().max(1).values * feat[torch.arange(self.in_channels), feat.abs().argmax(1)].sign()
            features.append(feat)
        features = torch.vstack(features).T

        return
