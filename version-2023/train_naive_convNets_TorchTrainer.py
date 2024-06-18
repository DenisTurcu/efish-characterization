import time
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from NaiveConvNet import NaiveConvNet, TwoPathsNaiveConvNet


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: Dataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        input_noise_amount: float,
        printing: int,
        num_plotting_samples: int,
        batch_size: int,
        loss_fn=F.mse_loss,
    ) -> None:
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.input_noise_amount = input_noise_amount
        self.save_every = save_every
        self.printing = printing
        self.num_plotting_samples = num_plotting_samples
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.init_train_loader()
        self.optimizer = optimizer
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def init_train_loader(self):
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(self.train_data),
        )
        self.loss_lambda = torch.zeros(len(self.train_data.wanted_predictions)).to(self.gpu_id)  # type: ignore
        for i, k in enumerate(self.train_data.wanted_predictions):  # type: ignore
            self.loss_lambda[i] = self.train_data.wanted_predictions[k][0]  # type: ignore

    def _run_epoch(self, epoch: int, data_id: int = 0):
        start_time = time.time()
        verbose = self.gpu_id == 0 and ((epoch + 1) % self.printing == 0 or epoch == 0)  # and data_id == 0
        if verbose:
            if data_id == 0:
                print(f"Epoch: {epoch+1}. GPU: {self.gpu_id}. Data ID: {str(data_id).rjust(2)}. Batch:", end=" ")
            else:
                print(f"                                      Data ID: {str(data_id).rjust(2)}. Batch:", end=" ")

        self.train_loader.sampler.set_epoch(epoch)  # type: ignore
        if self.gpu_id == 0:
            st_time1 = time.time()
        for i, (source, targets) in enumerate(self.train_loader):
            if self.gpu_id == 0:
                st_time2 = time.time()
            # source = (source * (1 + torch.randn(*source.shape) * self.input_noise_amount)).to(self.gpu_id)  # multiplicative noise
            source = (source + torch.randn(*source.shape) * self.input_noise_amount).to(self.gpu_id)  # additive noise
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets, epoch, data_id)
            if self.gpu_id == 0:
                end_time = time.time()
            if verbose:
                print(
                    f"{i+1}/{len(self.train_loader)} {st_time2 - st_time1:.2f} + {end_time - st_time2:.2f}s | ",  # type: ignore
                    end="",
                )
            if self.gpu_id == 0:
                st_time1 = end_time

        # adjust individual variable losses by their loss coefficient
        if self.gpu_id == 0 and data_id == 0:
            for k in self.train_data.wanted_predictions:  # type: ignore
                self.train_data.wanted_predictions[k][1][epoch] = np.sqrt(  # type: ignore
                    self.train_data.wanted_predictions[k][1][epoch] / self.train_data.size  # type: ignore
                )

        if verbose:
            print(f"row time: {time.time() - start_time:.1f}s.")

    def _run_batch(self, source, targets, epoch: int, data_id: int = 0):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output * self.loss_lambda, targets * self.loss_lambda)
        loss.backward()
        self.optimizer.step()

        # compute errors for each variable predicted by the model
        if self.gpu_id == 0 and data_id == 0:
            with torch.no_grad():
                for i, k in enumerate(self.train_data.wanted_predictions):  # type: ignore
                    self.train_data.wanted_predictions[k][1][epoch] += self.loss_fn(  # type: ignore
                        output[:, i], targets[:, i]
                    ).item() * len(source)

    def _save_checkpoint(self, epoch: int, fname: str = "train_naive_convNets_results/checkpoint"):
        ##################################################################
        # save model and plot ############################################
        ##################################################################
        start_time = time.time()
        self.model.eval()  # set model to eval mode for saving and testing purposes
        torch.save(self.model.module.state_dict(), f"{fname}_Epoch{epoch+1}_state_dict.pt")  # type: ignore

        in_data, out_data = next(iter(self.train_loader))
        idx = np.random.permutation(len(in_data))[: self.num_plotting_samples]
        with torch.no_grad():
            out_pred = self.model(in_data).detach().cpu().numpy()[idx]
        out_data = out_data[idx].detach().cpu().numpy()
        self.model.train()  # set model back to train mode

        # plot losses
        fig = plt.figure(figsize=(9, 6))
        for k in self.train_data.wanted_predictions:  # type: ignore
            plt.plot(
                self.train_data.wanted_predictions[k][1] * self.train_data.stats_predictions[k]["std"],  # type: ignore
                lw=4,
                label=k,
            )
        plt.legend(loc=3)
        plt.yscale("log")
        sns.despine(offset=0, trim=True)
        plt.savefig(f"{fname}_Epoch{epoch+1}_loss.png", dpi=200)
        plt.close(fig)

        # plot predictions vs true values
        feature_names = ["position_xs", "position_ys", "position_zs", "radii", "resistances", "capacitances"]
        true_vals = np.array(
            [
                out_data[:, i] * self.train_data.stats_predictions[k]["std"]  # type: ignore
                + self.train_data.stats_predictions[k]["mean"]  # type: ignore
                for i, k in enumerate(self.train_data.wanted_predictions)  # type: ignore
            ]
        ).T
        pred_vals = np.array(
            [
                out_pred[:, i] * self.train_data.stats_predictions[k]["std"]  # type: ignore
                + self.train_data.stats_predictions[k]["mean"]  # type: ignore
                for i, k in enumerate(self.train_data.wanted_predictions)  # type: ignore
            ]
        ).T
        fig, ax = plt.subplots(1, true_vals.shape[-1], figsize=(2 * true_vals.shape[-1], 2))
        for i in range(true_vals.shape[-1]):
            true_vs = true_vals[:, i]
            pred_vs = pred_vals[:, i]
            ax[i].scatter(true_vs, pred_vs, c="k", s=1, alpha=0.5, marker=".")
            ax[i].plot([true_vs.min(), true_vs.max()], [true_vs.min(), true_vs.max()], ls="--", c="k", lw=0.5)
            ax[i].set_xlabel("True", fontsize=8)
            if i == 0:
                ax[i].set_ylabel("Predicted", fontsize=8)
            ax[i].set_title(f"{feature_names[i]}\n$R^2$ = {r2_score(true_vs, pred_vs):.3f}", fontsize=10)
            ax[i].tick_params(axis="both", which="major", labelsize=6)
            ax[i].axis("equal")
            sns.despine(ax=ax[i], offset=0, trim=True)
        plt.tight_layout()
        plt.savefig(f"{fname}_Epoch{epoch+1}_predictions.png", dpi=200)
        # plt.savefig(f"{fname}_Epoch{epoch+1}_predictions.svg")
        plt.close(fig)

        # plot the first layer spatial convolutional filters
        if isinstance(self.model.module, NaiveConvNet):
            filters = self.model.module.sequence.conv1.conv.weight.detach().cpu().numpy()  # type: ignore
        elif isinstance(self.model.module, TwoPathsNaiveConvNet):
            filters = (
                torch.cat(
                    [
                        self.model.module.conv_MZ.conv1.conv.weight,  # type: ignore
                        self.model.module.conv_DLZ.conv1.conv.weight,  # type: ignore
                    ],
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
        else:
            raise TypeError("The model could not be understood.")
        vval = np.max(np.abs(filters))
        fig, ax = plt.subplots(filters.shape[1], filters.shape[0], figsize=(2 * filters.shape[0], 2 * filters.shape[1]))
        for i in range(filters.shape[1]):
            for j in range(filters.shape[0]):
                ax[i, j].imshow(filters[j, i], cmap="seismic", vmin=-vval, vmax=vval)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                sns.despine(ax=ax[i, j], left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(f"{fname}_Epoch{epoch+1}_filters.png", dpi=200)
        # plt.savefig(f"{fname}_Epoch{epoch+1}_filters.svg")
        plt.close(fig)

        end_time = time.time()
        print(f"Plotting and saving time: {end_time - start_time:.1f}s", end=", ")

    def train(
        self,
        max_epochs: int,
        N_times_load_new_data: int = 1,
        start_epoch: int = 0,
        checkpoint_fname: str = "train_naive_convNets_results/checkpoint",
    ):
        if self.gpu_id == 0:
            for k in self.train_data.wanted_predictions:  # type: ignore
                self.train_data.wanted_predictions[k][1] = np.zeros(max_epochs) * np.nan  # type: ignore

        for epoch in range(start_epoch, max_epochs):
            for data_id in range(N_times_load_new_data):
                # prepare losses for storing and live-plotting
                if self.gpu_id == 0 and data_id == 0:
                    for k in self.train_data.wanted_predictions:  # type: ignore
                        self.train_data.wanted_predictions[k][1][epoch] = 0  # type: ignore

                # run epoch training
                self._run_epoch(epoch, data_id)

                # load new data to RAM
                if (epoch + 1) % N_times_load_new_data == 0:
                    self.train_data.renew_loaded_data(  # type: ignore
                        verbose=(self.gpu_id == 0 and data_id == 0),
                        gpu_id=self.gpu_id,
                    )
                    self.init_train_loader()

            # save checkpoint
            if self.gpu_id == 0 and (epoch == 0 or (epoch + 1) % self.save_every == 0):
                self._save_checkpoint(epoch=epoch, fname=checkpoint_fname)

            if self.gpu_id == 0 and ((epoch + 1) % self.printing == 0 or epoch == 0):
                print()
