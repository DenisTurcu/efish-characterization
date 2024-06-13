import torch.nn as nn
from collections import OrderedDict
from helpers_conv_nn_models import compile_conv_layer, compile_fc_layer


class EndToEndConvNN(nn.Module):

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
                    dict(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=5,
                        stride=1,
                    ),
                ),
                (
                    "conv4",
                    dict(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=5,
                        stride=1,
                    ),
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
        super(EndToEndConvNN, self).__init__()

        # compile the network's layers
        self.sequence = OrderedDict()
        for key, layer in layers_properties.items():
            if "conv" in key:
                self.sequence[key] = compile_conv_layer(layer, activation)
            elif "fc" in key:
                self.sequence[key] = compile_fc_layer(layer, activation)
            else:
                raise ValueError(f'Layer type "{key}" from "layers_properties" is not compatible.')
        self.sequence = nn.Sequential(self.sequence)

    def forward(self, x):
        return self.sequence(x)  # type: ignore

    def forward_print_dims(self, x):
        out = x
        print(out.shape)
        for sequence in self.sequence:
            print("==================================================================================================")
            for layer in sequence:  # type: ignore
                print(layer)
                out = layer(out)
                print(out.shape)
                print(
                    "--------------------------------------------------------------------------------------------------"
                )
        return out
