import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ElectricPropertiesNN(nn.Module):
    def __init__(
        self,
        kernel_size: int = 7,
        in_channels: int = 2,
        poly_degree_distance: int = 4,
        poly_degree_radius: int = 3,
        activation: nn.Module = nn.Tanh(),
    ):
        super(ElectricPropertiesNN, self).__init__()
        self.in_channels = in_channels
        self.average_pooling_kernel_size = kernel_size
        self.average_pooling_kernel = nn.Parameter(
            torch.ones([in_channels, 1, kernel_size, kernel_size])
            / torch.ones([in_channels, 1, kernel_size, kernel_size]).sum(),
            requires_grad=False,
        )
        self.poly_degree_distance = poly_degree_distance
        self.poly_coeffs_distance = nn.Parameter(torch.randn(poly_degree_distance + 1), requires_grad=True)
        self.poly_degree_radius = poly_degree_radius
        self.poly_coeffs_radius = nn.Parameter(torch.randn(poly_degree_radius + 1), requires_grad=True)

        self.sequence = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 10),
            activation,
            nn.Linear(10, 20),
            activation,
            nn.Linear(20, 10),
            activation,
            nn.Linear(10, 2),
        )

    def forward(self, electric_images, distances, radii, return_features_and_multiplier=False):
        # apply spatial average pooling
        processed = nn.functional.conv2d(electric_images, self.average_pooling_kernel, groups=self.in_channels)

        # extract features from the two channels for MZ  and DLZ
        features = []
        for i in range(self.in_channels):
            feat = processed[:, i].reshape(electric_images.shape[0], -1)
            feat = feat.abs().max(1).values * feat[torch.arange(electric_images.shape[0]), feat.abs().argmax(1)].sign()
            features.append(feat)
        features = torch.vstack(features).T

        # compute and apply scale multiplier
        scale_multiplier_distance = self.compute_scale(distances, self.poly_coeffs_distance, self.poly_degree_distance)
        scale_multiplier_radius = self.compute_scale(radii, self.poly_coeffs_radius, self.poly_degree_radius)
        features = features * (scale_multiplier_distance * scale_multiplier_radius)[:, np.newaxis]

        # run NN model
        if return_features_and_multiplier:
            return self.sequence(features), features, scale_multiplier_distance, scale_multiplier_radius
        return self.sequence(features)

    def compute_scale(self, variable, poly_coeffs, poly_degree):
        variable = torch.pow(variable[:, np.newaxis], torch.arange(poly_degree + 1).to(variable.device))
        scale_multiplier = F.softplus((poly_coeffs * variable).sum(-1))
        return scale_multiplier
