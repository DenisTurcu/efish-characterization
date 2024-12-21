import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from ElectricPropertiesNN_PL import ElectricPropertiesNN_PL

import sys

sys.path.append("../../efish-physics-model/objects")
sys.path.append("../../efish-physics-model/helper_functions")
sys.path.append("../../efish-physics-model/uniform_points_generation")
sys.path.append("../end-to-end")

# from electric_properties_models_dataset import ElectricPropertiesModelsDataset  # noqa: E402
from electric_images_dataset import ElectricImagesDataset  # noqa: E402


def my_parser():
    parser = argparse.ArgumentParser(description="Train end-to-end convolutional neural network models.")
    parser.add_argument(
        "--data_dir_name",
        type=str,
        default="../../efish-physics-model/data/processed/data-2024_06_18-characterization_dataset",
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50_000,
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
        "--input_noise_type",
        type=str,
        default="additive",
        help="Type of noise to include to the input data: `additive` or `multiplicative`.",
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
    parser.add_argument(
        "--average_pooling_kernel_size",
        type=int,
        default=7,
        help="Size of the kernel for the average pooling layer.",
    )
    parser.add_argument(
        "--poly_degree_distance",
        type=int,
        default=4,
        help="Degree of the polynomial to use for the scale multiplier with object distance.",
    )
    parser.add_argument(
        "--poly_degree_radius",
        type=int,
        default=3,
        help="Degree of the polynomial to use for the scale multiplier with object size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = my_parser()

    original_dset = ElectricImagesDataset(
        data_dir_name="../../efish-physics-model/data/processed/data-2024_06_18-characterization_dataset",
        fish_t=args.fish_t,
        fish_u=args.fish_u,
    )
    original_stats = pd.DataFrame.from_dict(original_dset.worms_properties_stats)[
        ["position_xs", "position_ys", "position_zs", "radii", "resistances", "capacitances"]
    ]

    dset = ElectricImagesDataset(data_dir_name=args.data_dir_name, fish_t=args.fish_t, fish_u=args.fish_u)
    new_stats = pd.DataFrame.from_dict(dset.worms_properties_stats)[
        ["position_xs", "position_ys", "position_zs", "radii", "resistances", "capacitances"]
    ]
    dset.worms_properties_stats = original_dset.worms_properties_stats
    dset.worms_properties = (
        (
            dset.worms_properties[:] * new_stats.loc["std"].to_numpy()
            + new_stats.loc["mean"].to_numpy()
            - original_stats.loc["mean"].to_numpy()
        )
        / original_stats.loc["std"].to_numpy()
    ).astype(np.float32)

    model_PL = ElectricPropertiesNN_PL(
        kernel_size=args.average_pooling_kernel_size,
        poly_degree_distance=args.poly_degree_distance,
        poly_degree_radius=args.poly_degree_radius,
        in_channels=2,
        activation=args.activation,
        input_noise_std=args.input_noise_std,
        input_noise_type=args.input_noise_type,
    )

    # data loaders
    train_dset, valid_dset = torch.utils.data.random_split(dset, [0.85, 0.15])  # type: ignore
    # train_dset, valid_dset, _ = torch.utils.data.random_split(dset, [0.05, 0.05, 0.9])  # type: ignore
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=12)
    valid_loader = DataLoader(valid_dset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=12)

    logger = pl_loggers.TensorBoardLogger(save_dir=args.save_dir)
    trainer = L.Trainer(max_epochs=args.max_epochs, logger=logger, devices=[args.gpu])
    trainer.fit(model=model_PL, train_dataloaders=train_loader, val_dataloaders=valid_loader)
