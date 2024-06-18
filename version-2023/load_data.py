import dill
import numpy as np
import h5py
import os
import torch
import sys

sys.path.append("../../../electric_fish/ActiveZone/electrodynamic/helper_functions")
sys.path.append("../../../electric_fish/ActiveZone/electrodynamic/objects")
sys.path.append("../../../electric_fish/ActiveZone/electrodynamic/uniform_points_generation")
# sys.path.append("../../efish-physics-model/helper_functions")
# sys.path.append("../../efish-physics-model/objects")
# sys.path.append("../../efish-physics-model/uniform_points_generation")


def load_data_full(file_name, use_torch=True, find_base_id_for_each_EI=True, verbose=False):
    # load dataset
    dataset = dill.load(open(f"{file_name}.pkl", "rb"))
    print("Data loaded.")

    # gather electric images
    # base
    base_EI = np.array(dataset["electric_images"]["base"]["data"])
    base_LEODs = None
    if "LEODs" in dataset["electric_images"]["base"] and dataset["electric_images"]["base"]["LEODs"]:
        base_LEODs = np.array(dataset["electric_images"]["base"]["LEODs"])
    water_conductivities = dataset["aqua"]["conductivity"]
    boundary_normals = dataset["aqua"]["boundary_normals"]
    boundary_displacements = dataset["aqua"]["boundary_displacements"]
    if os.path.isfile(f"{file_name}.hdf5"):
        f = h5py.File(f"{file_name}.hdf5", "r")
        pert_EI = f["pert_EI"]
        pert_LEODs = None
        if os.path.isfile(f"{file_name}_LEODs.hdf5"):
            f = h5py.File(f"{file_name}_LEODs.hdf5", "r")
            pert_LEODs = f["pert_LEODs"]
    else:
        pert_EI = np.array(dataset["electric_images"]["pert"]["data"])
        pert_LEODs = None
        if "LEODs" in dataset["electric_images"]["pert"] and dataset["electric_images"]["base"]["LEODs"]:
            pert_LEODs = np.array(dataset["electric_images"]["pert"]["LEODs"])

    tail_lateral_angle = dataset["fish"]["bend_angle_lateral"]
    tail_dor_ven_angle = dataset["fish"]["bend_angle_dorso_ventral"]
    tail_location_percent = dataset["fish"]["bend_location_percentages"]
    fish_yaw = dataset["fish"]["fish_yaw"]
    fish_pitch = dataset["fish"]["fish_pitch"]
    fish_roll = dataset["fish"]["fish_roll"]
    # perturbed
    resistances = dataset["worms"]["resistances"]
    capacitances = dataset["worms"]["capacitances"]
    worm_radii = dataset["worms"]["radii"]
    worm_xs = dataset["worms"]["position_xs"]
    worm_ys = dataset["worms"]["position_ys"]
    worm_zs = dataset["worms"]["position_zs"]
    # statistics of responses by receptor
    receptors_avg_response = None
    receptors_std_response = None
    if "responses_avg" in dataset["electric_images"]:
        receptors_avg_response = dataset["electric_images"]["responses_avg"]
    if "responses_std" in dataset["electric_images"]:
        receptors_std_response = dataset["electric_images"]["responses_std"]
    # keep track of properties IDs
    properties_dict = {
        "water_conductivity": 0,
        "boundary_normals": 1,
        "boundary_displacements": 2,
        "tail_lateral_angle": 3,
        "tail_dor_ven_angle": 4,
        "tail_location_percent": 5,
        "fish_yaw": 6,
        "fish_pitch": 7,
        "fish_roll": 8,
        "resistances": 9,
        "capacitances": 10,
        "worm_radii": 11,
        "worm_xs": 12,
        "worm_ys": 13,
        "worm_zs": 14,
        "pert_EI": 15,
        "base_EI": 16,
    }

    # gather corresponding electric properties of the object, and water conductivity
    aqua_ids = dataset["aqua"]["combined_properties"]
    fish_ids = dataset["fish"]["combined_properties"]
    worm_ids = dataset["worms"]["combined_properties"]

    Nprop_aq = aqua_ids.shape[1]
    Nprop_fi = fish_ids.shape[1]
    Nprop_wo = worm_ids.shape[1]

    obj_ids_base = np.array(dataset["electric_images"]["base"]["objs_ids"])
    properties_ids_base = np.zeros((obj_ids_base.shape[0], Nprop_aq + Nprop_fi + 1)) * np.nan
    properties_ids_base[:, 0:Nprop_aq] = aqua_ids[
        obj_ids_base[:, dataset["electric_images"]["properties_dict"]["aqua"]]
    ]
    properties_ids_base[:, Nprop_aq : Nprop_aq + Nprop_fi] = fish_ids[  # noqa: E203
        obj_ids_base[:, dataset["electric_images"]["properties_dict"]["fish"]]
    ]
    properties_ids_base[:, -1] = np.arange(
        base_EI.shape[0]
    )  # base el image ID for reference and tracking during shuffling
    properties_ids_base = properties_ids_base.astype(int)

    obj_ids_pert = np.array(dataset["electric_images"]["pert"]["objs_ids"])
    properties_ids_pert = np.zeros((obj_ids_pert.shape[0], Nprop_aq + Nprop_fi + Nprop_wo + 2)) * np.nan
    properties_ids_pert[:, 0:Nprop_aq] = aqua_ids[
        obj_ids_pert[:, dataset["electric_images"]["properties_dict"]["aqua"]]
    ]
    properties_ids_pert[:, Nprop_aq : Nprop_aq + Nprop_fi] = fish_ids[  # noqa: E203
        obj_ids_pert[:, dataset["electric_images"]["properties_dict"]["fish"]]
    ]
    properties_ids_pert[:, Nprop_aq + Nprop_fi : Nprop_aq + Nprop_fi + Nprop_wo] = worm_ids[  # noqa: E203
        obj_ids_pert[:, dataset["electric_images"]["properties_dict"]["worm"]]
    ]
    properties_ids_pert[:, -2] = np.arange(
        pert_EI.shape[0]  # type: ignore
    )  # pert el image ID for reference and tracking during shuffling
    if find_base_id_for_each_EI:
        properties_ids_pert[:, -1] = (
            (
                properties_ids_base[np.newaxis, :, :-1]
                == properties_ids_pert[:, np.newaxis, : properties_ids_base.shape[1] - 1]
            )
            .sum(-1)
            .argmax(1)
        )
    else:
        properties_ids_pert[:, -1] = 0
    properties_ids_pert = properties_ids_pert.astype(int)

    ####################
    # Convert to Torch #
    ####################
    if use_torch:
        properties_ids_base = torch.tensor(properties_ids_base)
        properties_ids_pert = torch.tensor(properties_ids_pert)
        base_EI = torch.tensor(base_EI)
        base_LEODs = None if base_LEODs is None else torch.tensor(base_LEODs)
        pert_EI = torch.tensor(pert_EI)
        pert_LEODs = None if pert_LEODs is None else torch.tensor(pert_LEODs)
        water_conductivities = torch.tensor(water_conductivities)
        boundary_normals = torch.tensor(boundary_normals)
        boundary_displacements = [torch.tensor(x) for x in boundary_displacements]
        tail_lateral_angle = torch.tensor(tail_lateral_angle)
        tail_dor_ven_angle = torch.tensor(tail_dor_ven_angle)
        tail_location_percent = torch.tensor(tail_location_percent)
        fish_yaw = torch.tensor(fish_yaw)
        fish_pitch = torch.tensor(fish_pitch)
        fish_roll = torch.tensor(fish_roll)
        resistances = torch.tensor(resistances)
        capacitances = torch.tensor(capacitances)
        worm_radii = torch.tensor(worm_radii)
        worm_xs = torch.tensor(worm_xs)
        worm_ys = torch.tensor(worm_ys)
        worm_zs = torch.tensor(worm_zs)

    if verbose:
        print("Data extracted.")
        for k in properties_dict:
            print(f'     Component {properties_dict[k]}: {k.replace("_", " ")}')

    return (
        properties_dict,
        properties_ids_base,
        properties_ids_pert,
        base_EI,
        base_LEODs,
        pert_EI,
        pert_LEODs,
        water_conductivities,
        boundary_normals,
        boundary_displacements,
        tail_lateral_angle,
        tail_dor_ven_angle,
        tail_location_percent,
        fish_yaw,
        fish_pitch,
        fish_roll,
        resistances,
        capacitances,
        worm_radii,
        worm_xs,
        worm_ys,
        worm_zs,
        receptors_avg_response,
        receptors_std_response,
        dataset["aqua"]["aqua_objs"],
        dataset["fish"]["fish_objs"],
        dataset["worms"]["worm_objs"],
    )
