import argparse
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from collections import OrderedDict
from electric_images_dataset import ElectricImagesDataset
from EndToEndConvNN_PL import EndToEndConvNN_PL

import sys

sys.path.append("../../efish-physics-model/objects")
sys.path.append("../../efish-physics-model/helper_functions")
sys.path.append("../../efish-physics-model/uniform_points_generation")


def my_parser():
    parser = argparse.ArgumentParser(description="Train end-to-end convolutional neural network models.")
    parser.add_argument(
        "--data_dir_name",
        type=str,
        default="../../efish-physics-model/data/processed/data-2024_06_13-characterization_dataset",
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20_000,
        help="Batch size to use for training the model.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=3,
        help="GPU device to use for training the model.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of epochs to train the model.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save the trained model checkpoints",
    )
    parser.add_argument(
        "--input_noise_std",
        type=float,
        default=0.25,
        help="Standard deviation of the noise to add to the input data.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function to use in the model.",
    )
    parser.add_argument(
        "--fish_t",
        type=int,
        default=20,
        help="Number of fish receptors around the body of the fish (circular axis).",
    )
    parser.add_argument(
        "--fish_u",
        type=int,
        default=30,
        help="Number of fish receptors along the body of the fish (longitudinal axis).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = my_parser()

    dataset = pd.read_pickle(f"{args.data_dir_name}/dataset.pkl")
    h5py_file = h5py.File(f"{args.data_dir_name}/responses.hdf5", "r")["responses"]

    dset = ElectricImagesDataset(data_dir_name=args.data_dir_name, fish_t=args.fish_t, fish_u=args.fish_u)

    layers_properties = OrderedDict(
        [
            (
                "conv1",
                dict(in_channels=2, out_channels=4, kernel_size=7, stride=1, max_pool=dict(kernel_size=3, stride=1)),
            ),
            (
                "conv2",
                dict(in_channels=4, out_channels=16, kernel_size=5, stride=1),
            ),
            (
                "conv3",
                dict(in_channels=16, out_channels=8, kernel_size=3, stride=1),
            ),
            (
                "conv4",
                dict(in_channels=8, out_channels=4, kernel_size=3, stride=1, max_pool=dict(kernel_size=3, stride=1)),
            ),
            # the fully connected layers can have dropout or flatten layers - some can miss the activation
            ("fc1", dict(dropout=0.5, flatten=True, in_features=None, out_features=240)),
            ("fc2", dict(dropout=0.5, in_features=240, out_features=60)),
            ("fc3", dict(in_features=60, out_features=6, activation=False)),
        ]
    )

    model_PL = EndToEndConvNN_PL(
        layers_properties=layers_properties, activation=args.activation, input_noise_std=args.input_noise_std
    )

    # dummy forward pass to initialize the model
    dloader = DataLoader(dset, batch_size=4, shuffle=True)
    batch = next(iter(dloader))
    _ = model_PL.model(batch[0])

    # data loaders
    train_dset, valid_dset = torch.utils.data.random_split(dset, [0.85, 0.15])  # type: ignore
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=12)
    valid_loader = DataLoader(valid_dset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=12)

    logger = pl_loggers.TensorBoardLogger(save_dir=args.save_dir)
    trainer = L.Trainer(max_epochs=args.max_epochs, logger=logger, devices=[args.gpu])
    trainer.fit(model=model_PL, train_dataloaders=train_loader, val_dataloaders=valid_loader)
