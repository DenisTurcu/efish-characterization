import copy
import numpy as np
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
from helpers_conv_nn_models import make_true_vs_predicted_figure
from EndToEndConvNN import EndToEndConvNN
from EndToEndConvNN_TwoPaths import EndToEndConvNN2Paths


class EndToEndConvNN_PL(L.LightningModule):
    def __init__(
        self,
        layers_properties: dict,
        activation: str = "relu",
        input_noise_std: float = 0.5,
        input_noise_type: str = "additive",
        model_type: str = "regular",
        loss_lambda: list = [1, 1, 1, 1, 10, 10],
    ):
        """_summary_

        Args:
            layers_properties (dict): Dictionary containing the properties of the layers to be used in the model.
            activation (str, optional): Activation function. Defaults to "relu".
            input_noise_std (float, optional): Input noise std that helps regularize training. Defaults to 0.5.
            model_type (str, optional): "regular" for typical conv layer architecture. "two_paths" for separate
                processing of MZ and DLZ channels. Defaults to "regular".

        Raises:
            ValueError: If the activation function is not supported.
            ValueError: If the model type is not supported.
        """
        super(EndToEndConvNN_PL, self).__init__()

        if activation.lower() == "relu":
            model_activation = nn.ReLU()  # type: ignore
        elif activation.lower() == "tanh":
            model_activation = nn.Tanh()  # type: ignore
        else:
            raise ValueError(f"Activation {activation} not yet supported.")

        if model_type == "regular":
            self.model = EndToEndConvNN(
                layers_properties=copy.deepcopy(layers_properties),  # type: ignore
                activation=model_activation,  # type: ignore
            )
        elif model_type == "two_paths":
            self.model = EndToEndConvNN2Paths(
                layers_properties=copy.deepcopy(layers_properties),  # type: ignore
                activation=model_activation,  # type: ignore
            )
        else:
            raise ValueError(f"Model type {model_type} not yet supported.")
        assert (input_noise_type == "additive") or (input_noise_type == "multiplicative"), "Noise type not supported."
        self.input_noise_type = input_noise_type
        self.input_noise_std = input_noise_std
        self.number_outputs = layers_properties[next(reversed(layers_properties))]["out_features"]
        self.loss_lambda = torch.Tensor(loss_lambda[: self.number_outputs])
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, : self.number_outputs]
        # train with noise for regularization
        if self.input_noise_type == "additive":
            y_hat = self.model(x + torch.randn_like(x) * self.input_noise_std)
        elif self.input_noise_type == "multiplicative":
            y_hat = self.model(x * (1 + torch.randn_like(x) * self.input_noise_std))
        loss = nn.functional.mse_loss(y_hat * self.loss_lambda.to(y.device), y * self.loss_lambda.to(y.device))
        self.log("train_loss", nn.functional.mse_loss(y_hat, y))

        # setup for logging non-scalars, such as figures
        if (batch_idx % 100) == 0:
            tensorboard = self.logger.experiment  # type: ignore
            # log the first layer conv filters
            if "sequence" in next(iter(self.model.state_dict().keys())):
                filters = self.model.sequence.conv1.conv.weight.detach().cpu().numpy()  # type: ignore
            else:
                filters_MZ = self.model.conv_MZ.conv1.conv.weight.detach().cpu().numpy()  # type: ignore
                filters_DLZ = self.model.conv_DLZ.conv1.conv.weight.detach().cpu().numpy()  # type: ignore
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
            # log the training predictions as figure
            true_vals = y[:400].detach().cpu().numpy()
            pred_vals = y_hat[:400].detach().cpu().numpy()
            fig = make_true_vs_predicted_figure(true_vals, pred_vals)
            tensorboard.add_figure("train_predictions", fig, global_step=0, close=True)
            plt.close("all")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, : self.number_outputs]
        y_hat = self.model(x)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)

        # setup for logging non-scalars, such as figures
        if (batch_idx % 100) == 0 and self.global_step > 0:
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
