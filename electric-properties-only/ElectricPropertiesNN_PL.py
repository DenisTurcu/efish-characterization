import numpy as np
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("../end-to-end")

from helpers_conv_nn_models import make_true_vs_predicted_figure  # noqa: E402
from ElectricPropertiesNN import ElectricPropertiesNN  # noqa: E402


class ElectricPropertiesNN_PL(L.LightningModule):
    def __init__(
        self,
        kernel_size=7,
        in_channels=2,
        poly_degree_distance=4,
        poly_degree_radius=3,
        activation: str = "tanh",
        input_noise_std: float = 0.5,
        input_noise_type: str = "additive",
    ):
        super(ElectricPropertiesNN_PL, self).__init__()

        if activation.lower() == "relu":
            model_activation = nn.ReLU()  # type: ignore
        elif activation.lower() == "tanh":
            model_activation = nn.Tanh()  # type: ignore
        else:
            raise ValueError(f"Activation {activation} not yet supported.")
        self.model = ElectricPropertiesNN(
            kernel_size=kernel_size,
            in_channels=in_channels,
            poly_degree_distance=poly_degree_distance,
            poly_degree_radius=poly_degree_radius,
            activation=model_activation,
        )

        assert (input_noise_type == "additive") or (input_noise_type == "multiplicative"), "Noise type not supported."
        self.input_noise_type = input_noise_type
        self.input_noise_std = input_noise_std
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        distances = y[:, 1]  # extract distances
        radii = y[:, 3]  # extract radii
        y = y[:, 4:]  # keep electric properties only

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
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", nn.functional.mse_loss(y_hat, y))

        # setup for logging non-scalars, such as figures
        if (batch_idx % 100) == 0:
            tensorboard = self.logger.experiment  # type: ignore
            # log the 3D feature space of MZ responses, DLZ responses, and distance AND the scale function with distance
            fig = plt.figure(figsize=(12, 3))
            # plot the 3D feature space of MZ responses, DLZ responses, and distance
            ax3D = fig.add_subplot(131, projection="3d")
            xs = features[:, 0].detach().cpu().numpy()
            ys = features[:, 1].detach().cpu().numpy()
            distances = distances.detach().cpu().numpy()
            radii = radii.detach().cpu().numpy()
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
            ax_dist.scatter(distances, scale_multiplier_distance.detach().cpu().numpy(), c=distances, cmap="tab20")
            ax_dist.set_xlabel("Distances (normalized)", fontsize=10)
            ax_dist.set_ylabel("Scale multiplier", fontsize=12)
            sns.despine(ax=ax_dist, offset=5, trim=True)
            ax_rad = fig.add_subplot(133)
            ax_rad.scatter(radii, scale_multiplier_radius.detach().cpu().numpy(), c=radii, cmap="tab20")
            ax_rad.set_xlabel("Radii (normalized)", fontsize=10)
            ax_rad.set_ylabel("Scale multiplier", fontsize=12)
            sns.despine(ax=ax_rad, offset=5, trim=True)

            plt.tight_layout()
            tensorboard.add_figure("features-and-scales", fig, global_step=0, close=True)

            # log the training predictions as figure
            true_vals = y[:400].detach().cpu().numpy()
            pred_vals = y_hat[:400].detach().cpu().numpy()
            fig = make_true_vs_predicted_figure(true_vals, pred_vals, feature_names=["resistances", "capacitances"])
            tensorboard.add_figure("train_predictions", fig, global_step=0, close=True)
            plt.close("all")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        distances = y[:, 1]  # extract distances
        radii = y[:, 3]  # extract radii
        y = y[:, 4:]  # keep electric properties only
        y_hat = self.model(x, distances, radii)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)

        # setup for logging non-scalars, such as figures
        if (batch_idx % 100) == 0 and self.global_step > 0:
            tensorboard = self.logger.experiment  # type: ignore
            # log the training predictions
            idx = np.random.permutation(y.shape[0])[:400]
            true_vals = y[idx].detach().cpu().numpy()
            pred_vals = y_hat[idx].detach().cpu().numpy()
            fig = make_true_vs_predicted_figure(true_vals, pred_vals, feature_names=["resistances", "capacitances"])
            tensorboard.add_figure("valid_predictions", fig, global_step=0, close=True)
            plt.close("all")
        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        return optimizer
