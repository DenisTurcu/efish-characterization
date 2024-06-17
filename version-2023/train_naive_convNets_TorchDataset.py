import numpy as np
import h5py
import time
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from load_data import load_data_full  # type: ignore


class ElectricImagesDataset(Dataset):
    def __init__(
        self,
        file_name: str = (
            "/home/ssd2tb/dturcu/electric_fish_processed_data_2023/"
            "data_230324_YES_manyObjects_NO_boundaries_NO_tail_ONE_conductivity_OnlyFEW_R_C"
        ),
        include_electric_properties: bool = True,
        use_torch: bool = False,
        find_base_id_for_each_EI: bool = False,
        N_data_samples_that_fit_in_RAM: int = 400_000,
        receptors_grid: dict = dict(xx=20, yy=30, yy_retain=24),
        verbose: bool = False,
    ):
        self.include_electric_properties = include_electric_properties
        self.receptors_grid = receptors_grid
        self.read_data(
            file_name=file_name,
            use_torch=use_torch,
            find_base_id_for_each_EI=find_base_id_for_each_EI,
            verbose=verbose,
        )
        self.max_num_samples = N_data_samples_that_fit_in_RAM
        self.renew_loaded_data()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return (
            torch.Tensor(self.IMG_pert_EI_loaded[index,:,:,:self.receptors_grid["yy_retain"]] / self.base_stim[:,:,:self.receptors_grid["yy_retain"]]),  # type: ignore
            torch.Tensor(
                np.stack(
                    [
                        self.wanted_predictions[k][-1][  # type: ignore
                            self.properties_ids_pert_loaded[
                                index,
                                self.properties_dict[k],  # type: ignore
                            ]
                        ]
                        for k in self.wanted_predictions
                    ]
                )
            ),
        )

    def renew_loaded_data(self, verbose: bool = False, gpu_id: int = -1):
        if verbose:
            print("Loading new training data.", end=" ")
        start_time = time.time()
        del self.IMG_pert_EI_loaded
        self.start_rand_id = np.random.randint(self.IMG_pert_EI_shuf.shape[0] - self.max_num_samples)  # type: ignore
        self.IMG_pert_EI_loaded = self.IMG_pert_EI_shuf[  # type: ignore
            self.start_rand_id : self.start_rand_id + self.max_num_samples  # noqa: E203
        ].astype(np.float32)  # type: ignore
        idx_data_loaded = self.inverse_perm_shuf[
            self.start_rand_id : self.start_rand_id + self.max_num_samples  # noqa: E203
        ]
        self.properties_ids_pert_loaded = self.properties_ids_pert[idx_data_loaded]
        self.size = self.max_num_samples
        if verbose:
            print(f"Loaded in {time.time() - start_time:.1f} s.", end="")

    def read_data(
        self, file_name: str, use_torch: bool = False, find_base_id_for_each_EI: bool = False, verbose: bool = False
    ):
        [
            self.properties_dict,
            self.properties_ids_base,
            self.properties_ids_pert,
            self.base_EI,
            self.base_LEODs,
            self.pert_EI,
            self.pert_LEODs,
            # aquarium properties
            self.water_conductivities,
            self.boundary_normals,
            self.boundary_displacements,
            # fish properties
            self.tail_lateral_angle,
            self.tail_dor_ven_angle,
            self.tail_location_percent,
            self.fish_yaw,
            self.fish_pitch,
            self.fish_roll,
            # worm properties
            self.resistances,
            self.capacitances,
            self.worm_radii,
            self.worm_xs,
            self.worm_ys,
            self.worm_zs,
            # statistics of receptor responses
            self.receptors_avg,
            self.receptors_std,
            # objects used in simulations
            self.aqua_objs,
            self.fish_objs,
            self.worm_objs,
        ] = load_data_full(
            file_name=file_name, use_torch=use_torch, find_base_id_for_each_EI=find_base_id_for_each_EI, verbose=verbose
        )
        
        # setup the base electric image
        N_xx = self.receptors_grid["xx"]
        N_yy = self.receptors_grid["yy"]
        receptors = self.fish_objs[0].get_receptors_locations().copy()
        receptors[receptors[:,1]<0, 2] = self.fish_objs[0].get_vertical_semi_axis() * 2.2 - receptors[receptors[:,1]<0, 2]
        receptors = receptors[:,[0,2]]
        receptors_order = np.lexsort((receptors[:,0], receptors[:,1]))[::-1]
        self.base_stim = self.base_EI[0, receptors_order].reshape(N_xx, N_yy, -1).transpose(2,0,1)  # type: ignore
        
        # load the perturbed electric images
        f = h5py.File(f"{file_name}_IMGs.hdf5", "r")
        self.IMG_pert_EI = f["pert_EI"]
        g = h5py.File(file_name + "_IMGs_shuffled" + ".hdf5", "r")
        self.IMG_pert_EI_shuf = g["pert_EI"]  # do not load to memory as a whole
        self.rand_perm_shuf = g["rand_perm"][:].astype(int)  # type: ignore # load to memory
        self.inverse_perm_shuf = g["inverse_perm"][:].astype(int)  # type: ignore # load to memory

        ####################################################################################
        # make sure to adjust the "wanted_predictions" variable correctly  #################
        ####################################################################################
        if self.include_electric_properties:
            self.wanted_predictions = OrderedDict(
                # resistances  = [100, [], np.log10(self.resistances)-4],
                # capacitances = [100, [], -np.log10(self.capacitances)-8.5],
                worm_xs=[1, [], np.array(self.worm_xs) * 1e3],
                worm_ys=[1, [], np.array(self.worm_ys) * 1e3],
                worm_zs=[1, [], np.array(self.worm_zs) * 1e3],
                worm_radii=[1, [], np.array(self.worm_radii) * 1e3],
                resistances=[1, [], np.log10(self.resistances)],  # type: ignore
                capacitances=[1, [], np.log10(self.capacitances)],  # type: ignore
            )
        else:
            self.wanted_predictions = OrderedDict(
                worm_xs=[1, [], np.array(self.worm_xs) * 1e3],
                worm_ys=[1, [], np.array(self.worm_ys) * 1e3],
                worm_zs=[1, [], np.array(self.worm_zs) * 1e3],
                worm_radii=[1, [], np.array(self.worm_radii) * 1e3],
            )
        self.stats_predictions = OrderedDict()
        for k in self.wanted_predictions:
            self.stats_predictions[k] = dict(
                mean=self.wanted_predictions[k][-1].mean(), std=self.wanted_predictions[k][-1].std()  # type: ignore
            )
            self.wanted_predictions[k][-1] = (  # type: ignore
                self.wanted_predictions[k][-1] - self.stats_predictions[k]["mean"]  # type: ignore
            ) / self.stats_predictions[k]["std"]
        ####################################################################################

        self.IMG_pert_EI_loaded = None  # set as None such that it can be deleted on the first iteration as well
