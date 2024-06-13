import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset


class ElectricImagesDataset(Dataset):
    """Dataset class for the electric images dataset."""

    def __init__(
        self,
        data_dir_name="../../efish-physics-model/data/processed/data-2024_06_05-characterization_dataset",
        fish_t=20,
        fish_u=30,
    ):
        """Initialize the LFP response dataset.

        Args:

        """
        super(ElectricImagesDataset, self).__init__()
        self.data_dir_name = data_dir_name
        self.dataset = pd.read_pickle(f"{data_dir_name}/dataset.pkl")
        self.base_response = self.dataset["electric_images"]["base"]["responses"][0]
        self.responses = h5py.File(f"{data_dir_name}/responses.hdf5", "r")["responses"]
        self.fish_t = fish_t
        self.fish_u = fish_u

    def __len__(self):
        return self.responses.shape[0]  # type: ignore

    def __getitem__(self, index):
        response = self.responses[index]  # type: ignore
        response = response / self.base_response - 1
        response = response.reshape(self.fish_u, self.fish_t).T

        return (response,)
