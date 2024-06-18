import os
import sys
import argparse
import dill
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from NaiveConvNet import TwoPathsNaiveConvNet
from train_naive_convNets_TorchDataset import ElectricImagesDataset
from train_naive_convNets_TorchTrainer import Trainer

sys.path.append("../../../electric_fish/ActiveZone/electrodynamic/helper_functions")
sys.path.append("../../../electric_fish/ActiveZone/electrodynamic/objects")
sys.path.append("../../../electric_fish/ActiveZone/electrodynamic/uniform_points_generation")
# sys.path.append("../../efish-physics-model/helper_functions")
# sys.path.append("../../efish-physics-model/objects")
# sys.path.append("../../efish-physics-model/uniform_points_generation")


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def my_parser():
    parser = argparse.ArgumentParser(
        description="Parse command line arguments for training naive convNets on electric images IMGs."
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=(
            "/home/ssd2tb/dturcu/electric_fish_processed_data_2023/"
            "data_230324_YES_manyObjects_NO_boundaries_NO_tail_ONE_conductivity_OnlyFEW_R_C"
        ),
    )
    parser.add_argument("--N_epochs", type=int, default=10_000)
    parser.add_argument("--N_data_that_fits_in_RAM", type=int, default=400_000)
    parser.add_argument("--batch_size", type=int, default=20_000)
    parser.add_argument("--input_noise_amount", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--printing", type=int, default=1)
    parser.add_argument("--num_plotting_samples", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--pre_trained_model", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--include_electric_properties", type=bool, default=False)
    parser.add_argument("--checkpoint_fname", type=str, default="./trained_models/checkpoint")
    return parser.parse_args()


def process_model(layers_properties: OrderedDict, activation: str, fname: str = "", epoch: int = 0) -> nn.Module:
    if fname == "":
        dill.dump(
            dict(layers_properties=layers_properties, activation=activation),
            open(
                f"./hyperparams/{datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d-T-%H_%M_%S')}_hyperparams.pkl",
                "wb",
            ),
        )
        model = TwoPathsNaiveConvNet(
            layers_properties=layers_properties,
            activation=activation,
        )
    else:
        hyperparams = dill.load(open(f"{fname}_hyperparams.pkl", "rb"))
        state_dict = torch.load(f"{fname}_Epoch{epoch}_state_dict.pt", map_location="cpu")
        model = TwoPathsNaiveConvNet(**hyperparams)
        model.load_state_dict(state_dict)
        model.train()
    return model


def main(
    rank: int,
    world_size: int,
    save_every: int,
    total_epochs: int,
    batch_size: int,
    input_noise_amount: int,
    printing: int,
    num_plotting_samples: int,
    learning_rate: float,
    N_data_samples_that_fit_in_RAM: int,
    file_name: str,
    pre_trained_model: str,
    start_epoch: int,
    checkpoint_fname: str,
    include_electric_properties: bool,
):
    ddp_setup(rank, world_size)
    print(f"Start MAIN. Rank: {rank}.")
    dataset = ElectricImagesDataset(
        N_data_samples_that_fit_in_RAM=N_data_samples_that_fit_in_RAM,
        file_name=file_name,
        include_electric_properties=include_electric_properties,
    )
    number_outputs = len(dataset.wanted_predictions)
    model = process_model(
        layers_properties=OrderedDict(
            [
                (
                    "conv1",
                    dict(
                        in_channels=1, out_channels=4, kernel_size=7, stride=1, max_pool=dict(kernel_size=3, stride=1)
                    ),
                ),
                (
                    "conv2",
                    dict(in_channels=4, out_channels=16, kernel_size=5, stride=1),
                ),
                (
                    "conv3",
                    dict(
                        in_channels=16, out_channels=8, kernel_size=5, stride=1, max_pool=dict(kernel_size=3, stride=2)
                    ),
                ),
                ("fc1", dict(dropout=0.5, flatten=True, in_features=480, out_features=240)),
                ("fc2", dict(dropout=0.5, in_features=240, out_features=120)),
                ("fc3", dict(in_features=120, out_features=number_outputs, activation=False)),
            ]
        ),
        activation="relu",
        fname=pre_trained_model,
        epoch=start_epoch,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    trainer = Trainer(
        model=model,
        train_data=dataset,
        optimizer=optimizer,
        gpu_id=rank,
        save_every=save_every,
        input_noise_amount=input_noise_amount,
        printing=printing,
        num_plotting_samples=num_plotting_samples,
        batch_size=batch_size,
        loss_fn=F.mse_loss,
    )
    trainer.train(
        max_epochs=total_epochs,
        N_times_load_new_data=int(0.5 * dataset.IMG_pert_EI.shape[0] / N_data_samples_that_fit_in_RAM),  # type: ignore
        start_epoch=start_epoch,
        checkpoint_fname=checkpoint_fname,
    )
    destroy_process_group()


if __name__ == "__main__":
    args = my_parser()
    world_size = torch.cuda.device_count()
    mp.spawn(  # type: ignore
        main,
        args=(
            world_size,
            args.save_every,
            args.N_epochs,
            args.batch_size,
            args.input_noise_amount,
            args.printing,
            args.num_plotting_samples,
            args.learning_rate,
            args.N_data_that_fits_in_RAM,
            args.file_name,
            args.pre_trained_model,
            args.start_epoch,
            args.checkpoint_fname,
            args.include_electric_properties,
        ),
        nprocs=world_size,
    )
