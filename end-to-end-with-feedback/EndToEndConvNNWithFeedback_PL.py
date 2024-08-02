import copy
import numpy as np
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("../end-to-end")
sys.path.append("../electric-properties-only")

from helpers_conv_nn_models import make_true_vs_predicted_figure  # noqa: E402
from EndToEndConvNNWithFeedback import EndToEndConvNNWithFeedback  # noqa: E402


class EndToEndConvNNWithFeedback_PL(L.LightningModule):
    def __init__(
        self,
        # spatial model
        layers_properties: dict,
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
        input_noise_std: float = 0.25,
        input_noise_type: str = "additive",
        loss_lambda: list = [1, 2, 1, 4, 8, 8],
    ):
        super(EndToEndConvNNWithFeedback_PL, self).__init__()

        self.model = EndToEndConvNNWithFeedback(
            layers_properties=copy.deepcopy(layers_properties),
            activation_spatial=activation_spatial,
            model_type=model_type,
            kernel_size=kernel_size,
            in_channels=in_channels,
            poly_degree_distance=poly_degree_distance,
            poly_degree_radius=poly_degree_radius,
            activation_feedback=activation_feedback,
            use_estimates_as_feedback=use_estimates_as_feedback,
        )
        self.input_noise_type = input_noise_type
        self.input_noise_std = input_noise_std
        self.number_outputs = layers_properties[next(reversed(layers_properties))]["out_features"]
        self.loss_lambda = torch.Tensor(loss_lambda)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        distances = y[:, 1]  # extract distances
        radii = y[:, 3]  # extract radii

        # train with noise for regularization
        x_noise = torch.randn_like(x) * self.input_noise_std
        if self.input_noise_type == "additive":
            y_hat, features, scale_multiplier_distance, scale_multiplier_radius = self.model(
                x + x_noise, distances, radii, return_features_and_multiplier=True
            )
        elif self.input_noise_type == "multiplicative":
            y_hat, features, scale_multiplier_distance, scale_multiplier_radius = self.model(
                x * (1 + x_noise), distances, radii, return_features_and_multiplier=True
            )
        loss = nn.functional.mse_loss(y_hat * self.loss_lambda.to(y.device), y * self.loss_lambda.to(y.device))
        self.log("train_loss", nn.functional.mse_loss(y_hat, y))

        # setup for logging non-scalars, such as figures
        if (batch_idx % 100) == 0:
            if self.model.use_estimates_as_feedback:
                distances = y_hat[:, 1]
                radii = y_hat[:, 3]
            distances = distances.detach().cpu().numpy()
            radii = radii.detach().cpu().numpy()
            tensorboard = self.logger.experiment  # type: ignore
            # log the first layer conv filters
            if "sequence" in next(iter(self.model.spatial_model.state_dict().keys())):
                filters = self.model.spatial_model.sequence.conv1.conv.weight.detach().cpu().numpy()  # type: ignore
            else:
                filters_MZ = self.model.spatial_model.conv_MZ.conv1.conv.weight.detach().cpu().numpy()  # type: ignore
                filters_DLZ = self.model.spatial_model.conv_DLZ.conv1.conv.weight.detach().cpu().numpy()  # type: ignore
                filters = np.concatenate([filters_MZ, filters_DLZ], axis=1)

            vval = np.max(np.abs(filters))
            fig, ax = plt.subplots(
                filters.shape[1], filters.shape[0], figsize=(1.5 * filters.shape[0], 1.5 * filters.shape[1])
            )
            for i in range(filters.shape[1]):
                for j in range(filters.shape[0]):
                    ax[i, j].imshow(filters[j, i], cmap="seismic", vmin=-vval, vmax=vval)
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                    sns.despine(ax=ax[i, j], left=True, bottom=True)
            plt.tight_layout()
            tensorboard.add_figure("filters", fig, global_step=0, close=True)

            # log the 3D feature space of MZ responses, DLZ responses, and distance AND the scale function with distance
            fig = plt.figure(figsize=(12, 3))
            # plot the 3D feature space of MZ responses, DLZ responses, and distance
            ax3D = fig.add_subplot(131, projection="3d")
            xs = features[:, 0].detach().cpu().numpy()
            ys = features[:, 1].detach().cpu().numpy()
            ax3D.scatter(
                xs,
                ys,
                distances,
                c=distances,
                cmap="tab20",
                s=1,  # type: ignore
                marker=".",
            )
            ax3D.set_xlabel("MZ mod", fontsize=10)
            ax3D.set_ylabel("DLZ mod", fontsize=10)
            ax3D.set_zlabel("Distance (normalized)", fontsize=10)  # type: ignore
            # plot the scale function with distance
            ax_dist = fig.add_subplot(132)
            ax_dist.scatter(
                distances, scale_multiplier_distance.detach().cpu().numpy(), c=distances, cmap="tab20", s=2, alpha=0.3
            )
            ax_dist.set_xlabel("Distances (normalized)", fontsize=10)
            ax_dist.set_ylabel("Scale multiplier", fontsize=12)
            sns.despine(ax=ax_dist, offset=5, trim=True)
            ax_rad = fig.add_subplot(133)
            ax_rad.scatter(radii, scale_multiplier_radius.detach().cpu().numpy(), c=radii, cmap="tab20", s=2, alpha=0.3)
            ax_rad.set_xlabel("Radii (normalized)", fontsize=10)
            ax_rad.set_ylabel("Scale multiplier", fontsize=12)
            sns.despine(ax=ax_rad, offset=5, trim=True)
            plt.tight_layout()
            tensorboard.add_figure("features-and-scales", fig, global_step=0, close=True)

            # log the training predictions as figure
            true_vals = y[:400].detach().cpu().numpy()
            pred_vals = y_hat[:400].detach().cpu().numpy()
            fig = make_true_vs_predicted_figure(true_vals, pred_vals)
            tensorboard.add_figure("train_predictions", fig, global_step=0, close=True)
            plt.close("all")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        distances = y[:, 1]  # extract distances
        radii = y[:, 3]  # extract radii
        y_hat = self.model(x, distances, radii)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)

        # setup for logging non-scalars, such as figures
        if batch_idx == 0 and self.global_step > 0:
            tensorboard = self.logger.experiment  # type: ignore
            # log the training predictions
            idx = np.random.permutation(y.shape[0])[:400]
            true_vals = y[idx].detach().cpu().numpy()
            pred_vals = y_hat[idx].detach().cpu().numpy()
            fig = make_true_vs_predicted_figure(true_vals, pred_vals)
            tensorboard.add_figure("valid_predictions", fig, global_step=0, close=True)
            plt.close("all")
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
        return optimizer
