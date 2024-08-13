import dill
from torch.utils.data import Dataset


class ElectricPropertiesModelsDataset(Dataset):
    """Dataset class for the electric images dataset."""

    def __init__(
        self,
        data_dir_name="./",
        file_name="electric_properties_data.pkl",
    ):
        """Initialize the LFP response dataset.

        Args:

        """
        super(ElectricPropertiesModelsDataset, self).__init__()
        self.data_dir_name = data_dir_name
        self.modulations, self.worms_properties = dill.load(open(f"{data_dir_name}/{file_name}", "rb"))

    def __len__(self):
        return self.modulations.shape[0]  # type: ignore

    def __getitem__(self, index):
        return (self.modulations[index], self.worms_properties[index])  # type: ignore
