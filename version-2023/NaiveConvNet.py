import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# create a simple circular padding module for use use max_pool which does not implement circular padding by default.
class CircularPad2d(nn.Module):
    def __init__(self, padding_h, padding_w=0):
        super(CircularPad2d, self).__init__()
        self.padding = (padding_w, padding_w, padding_h, padding_h)

    def forward(self, x):
        return F.pad(x, self.padding, mode="circular")

    def extra_repr(self):
        return f'padding={self.padding} [e.g. (w,w,h,h,)], mode="circular"'


def compile_conv_layer(layer, activation):
    # extract max_pool layer if provided
    mp_layer = None
    if "max_pool" in layer:
        mp_layer = layer.pop("max_pool")
    # set up circular padding along one dimension only because fish is cylindrical
    layer["padding"] = (layer["kernel_size"] // 2, 0)
    layer["padding_mode"] = "circular"
    # make the conv layer's sequence
    conv_layer = OrderedDict(
        [
            ("conv", nn.Conv2d(**layer)),
            ("bn", nn.BatchNorm2d(layer["out_channels"])),
            ("fun", activation),
        ]
    )
    # include a max_pool layer if instructed
    if mp_layer is not None:
        conv_layer["circ_pad"] = CircularPad2d(mp_layer["kernel_size"] // 2)  # type: ignore
        conv_layer["max_pool"] = nn.MaxPool2d(**mp_layer)  # type: ignore
    return nn.Sequential(conv_layer)  # type: ignore


def compile_fc_layer(layer, activation):
    fc_layer = OrderedDict()
    # process activation
    include_activation = "activation" not in layer or ("activation" in layer and layer["activation"])
    if "activation" in layer:
        del layer["activation"]
    # flatten the input to this layer if instructed
    if "flatten" in layer and layer["flatten"]:
        fc_layer["flatten"] = nn.Flatten()
        del layer["flatten"]
    # dropout features if instructed
    if "dropout" in layer and layer["dropout"] is not None:
        assert layer["dropout"] > 0 and layer["dropout"] < 1, "Dropout must be a probability between 0 and 1"
        fc_layer["dropout"] = nn.Dropout(layer["dropout"])
        del layer["dropout"]
    # initialize the linear layer depending on whether in_features is known or not
    if layer["in_features"] is None:
        del layer["in_features"]
        fc_layer["linear"] = nn.LazyLinear(**layer)
    else:
        fc_layer["linear"] = nn.Linear(**layer)
    # include the activation only if instructed
    if include_activation:
        fc_layer["fun"] = activation
    return nn.Sequential(fc_layer)


class NaiveConvNet(nn.Module):
    def __init__(
        self,
        layers_properties=OrderedDict(
            [
                # conv layers have an appropriate batch_norm layer and circular padding because the fish is cylindrical
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
        activation="relu",
    ):
        super(NaiveConvNet, self).__init__()
        self.layers_properties = layers_properties
        self.activation = activation
        if activation == "relu":
            model_activation = nn.ReLU()
        elif activation == "tanh":
            model_activation = nn.Tanh()
        else:
            raise ValueError(f'Activation "{activation}" is not supported.')

        # compile the network's layers
        self.sequence = OrderedDict()
        for key, layer in layers_properties.items():
            if "conv" in key:
                self.sequence[key] = compile_conv_layer(layer, model_activation)
            elif "fc" in key:
                self.sequence[key] = compile_fc_layer(layer, model_activation)
            else:
                raise ValueError(f'Layer type "{key}" from "layers_properties" is not compatible.')
        self.sequence = nn.Sequential(self.sequence)

    def forward(self, x):
        return self.sequence(x)  # type: ignore

    def forward_print_dims(self, x):
        out = x
        print(out.shape)
        for seq in self.sequence:
            print("==================================================================================================")
            for layer in seq:  # type: ignore
                print(layer)
                out = layer(out)
                print(out.shape)
                print(
                    "--------------------------------------------------------------------------------------------------"
                )
        return out


class TwoPathsNaiveConvNet(nn.Module):
    def __init__(
        self,
        layers_properties=OrderedDict(
            [
                # conv layers have an appropriate batch_norm layer and circular padding because the fish is cylindrical
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
        activation="relu",
    ):
        super(TwoPathsNaiveConvNet, self).__init__()
        self.layers_properties = layers_properties
        self.activation = activation
        if activation == "relu":
            model_activation = nn.ReLU()
        elif activation == "tanh":
            model_activation = nn.Tanh()
        else:
            raise ValueError(f'Activation "{activation}" is not supported.')

        for key in layers_properties:
            if "conv" in key:
                continue
            elif "fc" in key:
                continue
            else:
                raise ValueError(f'Layer type "{key}" from "layers_properties" is not compatible.')

        # compile the network's conv layers
        self.conv_MZ = OrderedDict()
        self.conv_DLZ = OrderedDict()
        for key, layer in layers_properties.items():
            if "conv" in key:
                self.conv_MZ[key] = compile_conv_layer(copy.copy(layer), model_activation)
                self.conv_DLZ[key] = compile_conv_layer(copy.copy(layer), model_activation)
        self.conv_MZ = nn.Sequential(self.conv_MZ)
        self.conv_DLZ = nn.Sequential(self.conv_DLZ)

        # compile the networks's linear layers
        self.linear = OrderedDict()
        for key, layer in layers_properties.items():
            if "fc" in key:
                self.linear[key] = compile_fc_layer(layer, model_activation)
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
                    "--------------------------------------------------------------------------------------------------"
                )
        return out
