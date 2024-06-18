import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from collections import OrderedDict


def compile_conv_layer(layer: dict, activation: nn.Module) -> nn.Sequential:
    """Compile a convolutional layer based on the provided parameters.

    Args:
        layer (dict): Dictionary containing the parameters for the convolutional layer. Example below:
                conv_name -> dict(in_channels=1, out_channels=8, kernel_size=5, stride=1,
                                  max_pool=dict(kernel_size=3, stride=2))
        activation (nn.Module): Activation function to use after the convolutional layer.

    Returns:
        nn.Sequential: Torch nn.Sequential object representing the compiled layer.
    """
    max_pool_layer = None
    if "max_pool" in layer:
        max_pool_layer = layer.pop("max_pool")

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
    if max_pool_layer is not None:
        padding_size = max_pool_layer["kernel_size"] // 2
        conv_layer["circ_pad"] = nn.CircularPad2d((0, 0, padding_size, padding_size))  # type: ignore
        conv_layer["max_pool"] = nn.MaxPool2d(**max_pool_layer)  # type: ignore
    return nn.Sequential(conv_layer)  # type: ignore


def compile_fc_layer(layer: dict, activation: nn.Module) -> nn.Sequential:
    """Compile a fully-connected layer based on the provided parameters.

    Args:
        layer (dict): Dictionary containing the parameters for the convolutional layer. Example below:
                fully-connected_name -> dict(dropout=0.5, flatten=True, in_features=None, out_features=512)
        activation (nn.Module): Activation function to use after the convolutional layer.

    Returns:
        nn.Sequential: Torch nn.Sequential object representing the compiled layer.
    """
    fc_layer = OrderedDict()
    # process activation
    include_activation = ("activation" not in layer) or layer["activation"]
    if "activation" in layer:
        del layer["activation"]
    # flatten the input to this layer if instructed
    if "flatten" in layer and layer["flatten"]:
        fc_layer["flatten"] = nn.Flatten()
        del layer["flatten"]
    # dropout features if instructed
    if ("dropout" in layer) and (layer["dropout"] is not None):
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


def make_true_vs_predicted_figure(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    feature_names: list[str] = ["position_xs", "position_ys", "position_zs", "radii", "resistances", "capacitances"],
) -> plt.Figure:  # type: ignore
    """Create a figure comparing the true and predicted values for each feature.

    Args:
        true_vals (np.ndarray): True values.
        pred_vals (np.ndarray): Predicted values.
        feature_names (list[str], optional): Names of the features in the columns of the values.
            Defaults to ["position_xs", "position_ys", "position_zs", "radii", "resistances", "capacitances"].

    Returns:
        plt.figure.Figure: Figure object containing the comparison.
    """
    distance_id = feature_names.index("position_ys")
    scatter_colors = true_vals[:, distance_id]
    fig, ax = plt.subplots(1, true_vals.shape[-1], figsize=(2 * true_vals.shape[-1], 2))
    for i in range(true_vals.shape[-1]):
        true_vs = true_vals[:, i]
        pred_vs = pred_vals[:, i]
        ax[i].scatter(true_vs, pred_vs, c=scatter_colors, s=1, alpha=0.5, marker=".")
        ax[i].plot([true_vs.min(), true_vs.max()], [true_vs.min(), true_vs.max()], ls="--", c="k", lw=0.5)
        ax[i].set_xlabel("True", fontsize=8)
        if i == 0:
            ax[i].set_ylabel("Predicted", fontsize=8)
        ax[i].set_title(f"{feature_names[i]}\n$R^2$ = {r2_score(true_vs, pred_vs):.3f}", fontsize=10)
        ax[i].tick_params(axis="both", which="major", labelsize=6)
        ax[i].axis("equal")
        sns.despine(ax=ax[i], offset=0, trim=True)
    plt.tight_layout()
    return fig
