import torch
import torch.nn as nn
from collections import OrderedDict
import sys

sys.path.append("../end-to-end")
sys.path.append("../electric_properties_only")

from EndToEndConvNN import EndToEndConvNN  # noqa: E402
from EndToEndConvNN_TwoPaths import EndToEndConvNN2Paths  # noqa: E402
from ElectricPropertiesNN import ElectricPropertiesNN  # noqa: E402


class EndToEndConvNNWithFeedback(nn.Module):
    def __init__(
        self,
        # spatial model properties
        layers_properties: dict = OrderedDict(
            [
                (
                    "conv1",
                    dict(
                        in_channels=1, out_channels=8, kernel_size=5, stride=1, max_pool=dict(kernel_size=3, stride=2)
                    ),
                ),
                (
                    "conv2",
                    dict(
                        in_channels=8, out_channels=32, kernel_size=5, stride=1, max_pool=dict(kernel_size=3, stride=2)
                    ),
                ),
                (
                    "conv3",
                    dict(in_channels=32, out_channels=64, kernel_size=5, stride=1),
                ),
                (
                    "conv4",
                    dict(in_channels=64, out_channels=64, kernel_size=5, stride=1),
                ),
                (
                    "conv5",
                    dict(
                        in_channels=64, out_channels=32, kernel_size=5, stride=1, max_pool=dict(kernel_size=3, stride=2)
                    ),
                ),
                # the fully connected layers can have dropout or flatten layers - some can miss the activation
                ("fc1", dict(dropout=0.5, flatten=True, in_features=None, out_features=512)),
                ("fc2", dict(dropout=0.5, in_features=512, out_features=512)),
                ("fc3", dict(in_features=512, out_features=6, activation=False)),
            ]
        ),
        activation_spatial: str = "relu",
        model_type: str = "regular",
        # feedback model properties (for extracting electric properties)
        kernel_size: int = 7,
        in_channels: int = 2,
        poly_degree_distance: int = 4,
        poly_degree_radius: int = 3,
        activation_feedback: str = "relu",
        # miscellaneous properties
        use_estimates_as_feedback: bool = False,
    ):
        super(EndToEndConvNNWithFeedback, self).__init__()

        # spatial model
        if activation_spatial.lower() == "relu":
            spatial_activation = nn.ReLU()  # type: ignore
        elif activation_spatial.lower() == "tanh":
            spatial_activation = nn.Tanh()  # type: ignore
        else:
            raise ValueError(f"Activation {activation_spatial} not yet supported.")

        if model_type == "regular":
            self.spatial_model = EndToEndConvNN(
                layers_properties=layers_properties,  # type: ignore
                activation=spatial_activation,  # type: ignore
            )
        elif model_type == "two_paths":
            self.spatial_model = EndToEndConvNN2Paths(
                layers_properties=layers_properties,  # type: ignore
                activation=spatial_activation,  # type: ignore
            )
        else:
            raise ValueError(f"Model type {model_type} not yet supported.")

        # feedback model
        if activation_feedback.lower() == "relu":
            feedback_activation = nn.ReLU()  # type: ignore
        elif activation_feedback.lower() == "tanh":
            feedback_activation = nn.Tanh()  # type: ignore
        else:
            raise ValueError(f"Activation {activation_feedback} not yet supported.")

        self.feedback_model = ElectricPropertiesNN(
            kernel_size=kernel_size,
            in_channels=in_channels,
            poly_degree_distance=poly_degree_distance,
            poly_degree_radius=poly_degree_radius,
            activation=feedback_activation,
        )

        # miscellaneous properties
        self.use_estimates_as_feedback = use_estimates_as_feedback

    def forward(self, electric_images, distances=None, radii=None, return_features_and_multiplier=False):
        spatial_properties = self.spatial_model(electric_images)
        assert self.use_estimates_as_feedback or (
            distances is not None and radii is not None
        ), "Distances and radii must either be provided OR used from spatial model estimates."
        if self.use_estimates_as_feedback:
            distances = spatial_properties[:, 1]
            radii = spatial_properties[:, 2]
        electric_properties = self.feedback_model(electric_images, distances, radii, return_features_and_multiplier)
        if return_features_and_multiplier:
            return (
                torch.cat([spatial_properties, electric_properties[0]], dim=1),
                electric_properties[1],  # features
                electric_properties[2],  # scale multiplier distance
                electric_properties[3],  # scale multiplier radius
            )
        return torch.cat([spatial_properties, electric_properties], dim=1)
