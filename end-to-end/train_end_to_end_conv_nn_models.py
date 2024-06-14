import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from collections import OrderedDict
from electric_images_dataset import ElectricImagesDataset
from EndToEndConvNN import EndToEndConvNN
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

activation = "relu"
layers_properties = OrderedDict(
    [
        (
            "conv1",
            dict(in_channels=2, out_channels=8, kernel_size=5, stride=1, max_pool=dict(kernel_size=3, stride=1)),
        ),
        (
            "conv2",
            dict(in_channels=8, out_channels=32, kernel_size=5, stride=1, max_pool=dict(kernel_size=3, stride=1)),
        ),
        (
            "conv3",
            dict(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
        ),
        (
            "conv4",
            dict(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
        ),
        (
            "conv5",
            dict(in_channels=64, out_channels=32, kernel_size=3, stride=1, max_pool=dict(kernel_size=3, stride=2)),
        ),
        # the fully connected layers can have dropout or flatten layers - some can miss the activation
        ("fc1", dict(dropout=0.5, flatten=True, in_features=None, out_features=512)),
        ("fc2", dict(dropout=0.5, in_features=512, out_features=512)),
        ("fc3", dict(in_features=512, out_features=6, activation=False)),
    ]
)
if activation.lower() == "relu":
    activation = nn.ReLU()
elif activation.lower() == "tanh":
    activation = nn.Tanh()
else:
    raise ValueError(f"Activation {activation} not yet supported.")
model = EndToEndConvNN(layers_properties=layers_properties, activation=activation)  # type: ignore

# dummy forward pass to initialize the model
dloader = DataLoader(dset, batch_size=4, shuffle=True)
batch = next(iter(dloader))
_ = model(batch[0])

train_dset, valid_dset = torch.utils.data.random_split(dset, [0.8, 0.2])  # type: ignore
train_loader = DataLoader(train_dset, batch_size=5120, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dset, batch_size=5120, shuffle=False, drop_last=True)

model_PL = EndToEndConvNN_PL(model)

trainer = L.Trainer(max_steps=1000, devices=[3])
trainer.fit(model=model_PL, train_dataloaders=train_loader, val_dataloaders=valid_loader)
