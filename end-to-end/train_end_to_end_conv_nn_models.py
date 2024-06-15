import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader
import lightning as L
from collections import OrderedDict
from electric_images_dataset import ElectricImagesDataset
from EndToEndConvNN_PL import EndToEndConvNN_PL

import sys

sys.path.append("../../efish-physics-model/objects")
sys.path.append("../../efish-physics-model/helper_functions")
sys.path.append("../../efish-physics-model/uniform_points_generation")

data_dir_name = "../../efish-physics-model/data/processed/data-2024_06_13-characterization_dataset"
# data_dir_name = "../../efish-physics-model/data/processed/data-2024_06_13-characterization_dataset_mockup"
dataset = pd.read_pickle(f"{data_dir_name}/dataset.pkl")
h5py_file = h5py.File(f"{data_dir_name}/responses.hdf5", "r")["responses"]

dset = ElectricImagesDataset(data_dir_name=data_dir_name, fish_t=20, fish_u=30)

input_noise_std = 0.25
activation = "relu"
layers_properties = OrderedDict(
    [
        (
            "conv1",
            dict(in_channels=2, out_channels=4, kernel_size=7, stride=1, max_pool=dict(kernel_size=3, stride=1)),
        ),
        # (
        #     "conv2",
        #     dict(in_channels=4, out_channels=4, kernel_size=5, stride=1, max_pool=dict(kernel_size=3, stride=1)),
        # ),
        (
            "conv2",
            dict(
                in_channels=4,
                out_channels=16,
                kernel_size=5,
                stride=1,
            ),
        ),
        (
            "conv3",
            dict(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=1,
            ),
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
    layers_properties=layers_properties, activation=activation, input_noise_std=input_noise_std
)

# dummy forward pass to initialize the model
dloader = DataLoader(dset, batch_size=4, shuffle=True)
batch = next(iter(dloader))
_ = model_PL.model(batch[0])

train_dset, valid_dset, _ = torch.utils.data.random_split(dset, [0.05, 0.01, 0.94])  # type: ignore
train_loader = DataLoader(train_dset, batch_size=25_000, shuffle=True, drop_last=True, num_workers=12)
valid_loader = DataLoader(valid_dset, batch_size=25_000, shuffle=False, drop_last=True, num_workers=12)

trainer = L.Trainer(max_epochs=100, devices=[3])
trainer.fit(model=model_PL, train_dataloaders=train_loader, val_dataloaders=valid_loader)
