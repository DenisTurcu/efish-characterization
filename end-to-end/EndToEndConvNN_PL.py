import numpy as np
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns


class EndToEndConvNN_PL(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        tensorboard = self.logger.experiment
        filters = self.model.sequence.conv1.conv.weight.detach().cpu().numpy()
        vval = np.max(np.abs(filters))
        fig, ax = plt.subplots(filters.shape[1], filters.shape[0], figsize=(2 * filters.shape[0], 2 * filters.shape[1]))
        for i in range(filters.shape[1]):
            for j in range(filters.shape[0]):
                ax[i, j].imshow(filters[j, i], cmap="seismic", vmin=-vval, vmax=vval)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                sns.despine(ax=ax[i, j], left=True, bottom=True)
        plt.tight_layout()
        tensorboard.add_figure("filters", fig, global_step=self.global_step, close=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
