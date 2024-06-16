import copy
import torch
import torch.nn as nn
from collections import OrderedDict
from helpers_conv_nn_models import compile_conv_layer, compile_fc_layer


class EndToEndConvNN2Paths(nn.Module):
    def __init__(
        self,
        # the conv layers have an appropriate batch_norm layer and circular padding because the fish is cylindrical
        layers_properties=OrderedDict(
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
        activation=nn.ReLU(),
    ):
        super(EndToEndConvNN2Paths, self).__init__()

        # compile the network's conv layers
        self.conv_MZ = OrderedDict()
        self.conv_DLZ = OrderedDict()
        for key, layer in layers_properties.items():
            if "conv" in key:
                self.conv_MZ[key] = compile_conv_layer(copy.copy(layer), activation)
                self.conv_DLZ[key] = compile_conv_layer(copy.copy(layer), activation)
            elif "fc" in key:
                continue
            else:
                raise ValueError(f'Layer type "{key}" from "layers_properties" is not compatible.')
        self.conv_MZ = nn.Sequential(self.conv_MZ)
        self.conv_DLZ = nn.Sequential(self.conv_DLZ)

        # compile the networks's linear layers
        self.linear = OrderedDict()
        for key, layer in layers_properties.items():
            if "conv" in key:
                continue
            elif "fc" in key:
                self.linear[key] = compile_fc_layer(layer, activation)
            else:
                raise ValueError(f'Layer type "{key}" from "layers_properties" is not compatible.')
        self.linear = nn.Sequential(self.linear)

    def forward(self, x):
        x1 = self.conv_MZ(x[:, 0:1])  # type: ignore
        x2 = self.conv_DLZ(x[:, 1:2])  # type: ignore
        x = torch.cat([x1, x2], dim=1)
        return self.linear(x)  # type: ignore

    def forward_print_dims(self, x):
        out1 = x[:, 0:1]
        print(out1.shape)
        for conv in self.conv_MZ:
            print("==================================================================================================")
            for layer in conv:  # type: ignore
                print(layer)
                out1 = layer(out1)
                print(out1.shape)
                print(
                    "--------------------------------------------------------------------------------------------------"
                )
        out2 = self.conv_DLZ(x[:, 1:2])  # type: ignore
        out = torch.cat([out1, out2], dim=1)
        print("==================================================================================================")
        print(out.shape)
        for lin in self.linear:
            print("==================================================================================================")
            for layer in lin:  # type: ignore
                print(layer)
                out = layer(out)
                print(out.shape)
                print(
                    "-------------------------------------------------------------------------------------------------------"
                )
        return out
