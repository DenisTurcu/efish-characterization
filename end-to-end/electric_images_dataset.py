import dill
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
        dataset = pd.read_pickle(f"{data_dir_name}/dataset.pkl")
        self.process_worms_properties(dataset)
        self.base_response = dataset["electric_images"]["base"]["responses"][0]
        self.responses = h5py.File(f"{data_dir_name}/responses.hdf5", "r")["responses"]
        self.fish_t = fish_t
        self.fish_u = fish_u

    def __len__(self):
        return self.responses.shape[0]  # type: ignore

    def __getitem__(self, index):
        response = self.responses[index]  # type: ignore
        response = (response / self.base_response - 1) * 100
        response = response.reshape(self.fish_u, self.fish_t, 2).transpose(2, 1, 0).astype(np.float32)

        return (response, self.worms_properties[index])  # type: ignore

    def process_worms_properties(self, dataset):
        try:
            self.worms_properties = h5py.File(
                f"{self.data_dir_name}/torch_dataset_preprocess-worms_properties.hdf5", "r"
            )["worms_properties"]
            self.worms_properties_stats = dill.load(
                open(f"{self.data_dir_name}/torch_dataset_preprocess-worms_properties_stats.pkl", "rb")
            )
        except FileNotFoundError:
            worms_properties = dataset["worms"]["dataframe"]
            for k in worms_properties:
                worms_properties[k] = worms_properties[k].apply(lambda x: dataset["worms"][k][x])
            worms_properties["resistances"] = worms_properties["resistances"].apply(np.log10)
            worms_properties["capacitances"] = worms_properties["capacitances"].apply(np.log10)

            worms_properties_stats = {}
            for k in worms_properties:
                worms_properties_stats[k] = {
                    "mean": worms_properties[k].mean(),
                    "std": worms_properties[k].std() + 1e-15,
                }
                worms_properties[k] = (
                    worms_properties[k] - worms_properties_stats[k]["mean"]
                ) / worms_properties_stats[k]["std"]

            worms_properties = (
                worms_properties[["position_xs", "position_ys", "position_zs", "radii", "resistances", "capacitances"]]
                .to_numpy()
                .astype(np.float32)
            )

            with h5py.File(f"{self.data_dir_name}/torch_dataset_preprocess-worms_properties.hdf5", "w") as f:
                f.create_dataset("worms_properties", data=worms_properties, dtype=np.float32)
            dill.dump(
                worms_properties_stats,
                open(f"{self.data_dir_name}/torch_dataset_preprocess-worms_properties_stats.pkl", "wb"),
            )

            self.process_worms_properties(dataset)
