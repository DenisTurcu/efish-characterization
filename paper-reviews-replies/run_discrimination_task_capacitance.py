import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append("../electric-properties-only")
sys.path.append("../end-to-end")
sys.path.append("../end-to-end-with-feedback")
sys.path.append("../../efish-physics-model/objects")
sys.path.append("../../efish-physics-model/helper_functions")
sys.path.append("../../efish-physics-model/uniform_points_generation")

# from helpers_conv_nn_models import make_true_vs_predicted_figure
from electric_images_dataset import ElectricImagesDataset
from EndToEndConvNN_PL import EndToEndConvNN_PL
from EndToEndConvNNWithFeedback_PL import EndToEndConvNNWithFeedback_PL
from ElectricPropertiesNN_PL import ElectricPropertiesNN_PL


models = pd.DataFrame()


batch_size = 100
data_dir_name = "../../efish-physics-model/data/processed/data-2024_06_18-characterization_dataset"
original_dset = ElectricImagesDataset(data_dir_name=data_dir_name, fish_t=20, fish_u=30)
original_stats = pd.DataFrame.from_dict(original_dset.worms_properties_stats)[
    ["position_xs", "position_ys", "position_zs", "radii", "resistances", "capacitances"]
]
data_dir_name = "../../efish-physics-model/data/processed/data-2024_12_19-discrimination_dataset-capacitance"
raw_dataset = pd.read_pickle(f"{data_dir_name}/dataset.pkl")
dset = ElectricImagesDataset(data_dir_name=data_dir_name, fish_t=20, fish_u=30)
new_stats = pd.DataFrame.from_dict(dset.worms_properties_stats)[
    ["position_xs", "position_ys", "position_zs", "radii", "resistances", "capacitances"]
]
dset.worms_properties_stats = original_dset.worms_properties_stats
dset.worms_properties = (
    dset.worms_properties[:] * new_stats.loc["std"].to_numpy()
    + new_stats.loc["mean"].to_numpy()
    - original_stats.loc["mean"].to_numpy()
) / original_stats.loc["std"].to_numpy()
dloader = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=12)


for folder in list(np.sort(glob.glob("../figures/stats-panel/full-model*"))):
    checkpoint_path = f"{folder}/lightning_logs/version_0/checkpoints/epoch=4-step=25015.ckpt"
    model = EndToEndConvNN_PL.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    model.cpu()
    rand_seed = int(folder.split("_")[1].split("-")[0])
    lambda_RC = int(folder.split("_")[-1])
    models = pd.concat(
        [
            models,
            pd.DataFrame({"rand_seed": [rand_seed], "lambda_RC": [lambda_RC], "model_type": "full", "model": [model]}),
        ]
    ).reset_index(drop=True)


checkpoint_path = (
    f"../electric-properties-only/retrain_scale/lightning_logs/version_6/checkpoints/epoch=999-step=35000.ckpt"
)
retrained_scale_model = ElectricPropertiesNN_PL.load_from_checkpoint(checkpoint_path)
retrained_scale_model.eval()
retrained_scale_model.freeze()
retrained_scale_model.cpu()


for i, folder in enumerate(list(np.sort(glob.glob("../figures/stats-panel/feedback-with-values*")))):
    checkpoint_path = f"{folder}/lightning_logs/version_0/checkpoints/epoch=4-step=25015.ckpt"
    model = EndToEndConvNNWithFeedback_PL.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    model.cpu()
    model.model.spatial_model = models.loc[i, "model"].model
    # model.model.spatial_model.linear.fc3.linear.weight = torch.nn.Parameter(models.loc[i, "model"].model.linear.fc3.linear.weight[:4].clone())
    model.model.feedback_model = retrained_scale_model.model

    rand_seed = int(folder.split("_")[1].split("-")[0])
    lambda_RC = int(folder.split("_")[-1])
    models = pd.concat(
        [
            models,
            pd.DataFrame(
                {"rand_seed": [rand_seed], "lambda_RC": [lambda_RC], "model_type": "feedback_vals", "model": [model]}
            ),
        ]
    ).reset_index(drop=True)

for i, folder in enumerate(list(np.sort(glob.glob("../figures/stats-panel/feedback-with-estimates*")))):
    checkpoint_path = f"{folder}/lightning_logs/version_0/checkpoints/epoch=4-step=25015.ckpt"
    model = EndToEndConvNNWithFeedback_PL.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    model.cpu()
    model.model.spatial_model = models.loc[i, "model"].model
    # model.model.spatial_model.linear.fc3.linear.weight = torch.nn.Parameter(models.loc[i, "model"].model.linear.fc3.linear.weight[:4].clone())
    model.model.feedback_model = retrained_scale_model.model
    rand_seed = int(folder.split("_")[1].split("-")[0])
    lambda_RC = int(folder.split("_")[-1])
    models = pd.concat(
        [
            models,
            pd.DataFrame(
                {"rand_seed": [rand_seed], "lambda_RC": [lambda_RC], "model_type": "feedback_esti", "model": [model]}
            ),
        ]
    ).reset_index(drop=True)

models = models[models["lambda_RC"] > 0].reset_index(drop=True)


eis = next(iter(dloader))[0]
props = next(iter(dloader))[1].to(torch.float32)
discrimination_data = raw_dataset["worms"]["dataframe"].copy()
n_repeats = 100
noise_amount = 0.1


def train_capacitance_discrimination(
    dfrow, n_repeats=n_repeats, noise_amount=noise_amount, discrimination_data=discrimination_data, props=props, eis=eis
):
    print(dfrow["model_type"], dfrow["lambda_RC"], end=" | ")
    model = dfrow["model"]
    model_type = dfrow["model_type"]
    discrimination_performance = {}
    for pref_c_id in discrimination_data["capacitances"].unique():
        print(pref_c_id, end=" ")
        mask_sp = discrimination_data["capacitances"] == pref_c_id
        mask_sm = ~mask_sp
        eis_sp = eis[mask_sp]
        eis_sm = eis[mask_sm]
        train_eis_sp = torch.tile(eis_sp, (n_repeats * len(eis_sm), 1, 1, 1))
        train_eis_sm = torch.tile(eis_sm, (n_repeats, 1, 1, 1))
        train_eis_sp = train_eis_sp + torch.randn_like(train_eis_sp) * noise_amount
        train_eis_sm = train_eis_sm + torch.randn_like(train_eis_sm) * noise_amount
        if model_type == "feedback_vals":
            props_sp = props[mask_sp]
            props_sm = props[mask_sm]
            props_sp = torch.tile(props_sp, (n_repeats * len(eis_sm), 1))
            props_sm = torch.tile(props_sm, (n_repeats, 1))
            train_preds_sp = model.model(train_eis_sp, props_sp[:, 1], props_sp[:, 3]).detach().numpy()
            train_preds_sm = model.model(train_eis_sm, props_sm[:, 1], props_sm[:, 3]).detach().numpy()
        else:
            train_preds_sp = model.model(train_eis_sp).detach().numpy()
            train_preds_sm = model.model(train_eis_sm).detach().numpy()
        train_preds_mixed = np.stack([train_preds_sp, train_preds_sm])
        train_choices = np.random.rand(2, len(train_preds_sp)).argsort(0)
        train_preds_mixed = train_preds_mixed[train_choices, np.arange(len(train_preds_sp))]
        train_x = train_preds_mixed.transpose(1, 2, 0).reshape(len(train_preds_sp), -1)
        train_y = train_choices[1].astype(int)
        rfc = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1)
        rfc.fit(train_x, train_y)

        valid_eis_sm = torch.tile(eis, (n_repeats, 1, 1, 1))
        valid_eis_sp = torch.tile(eis[mask_sp], (len(valid_eis_sm), 1, 1, 1))
        valid_eis_sp = valid_eis_sp + torch.randn_like(valid_eis_sp) * noise_amount
        valid_eis_sm = valid_eis_sm + torch.randn_like(valid_eis_sm) * noise_amount
        if model_type == "feedback_vals":
            props_sp = props[mask_sp]
            props_sm = props
            props_sp = torch.tile(props_sp, (len(valid_eis_sm), 1))
            props_sm = torch.tile(props_sm, (n_repeats, 1))
            valid_preds_sp = model.model(valid_eis_sp, props_sp[:, 1], props_sp[:, 3]).detach().numpy()
            valid_preds_sm = model.model(valid_eis_sm, props_sm[:, 1], props_sm[:, 3]).detach().numpy()
        else:
            valid_preds_sm = model.model(valid_eis_sm).detach().numpy()
            valid_preds_sp = model.model(valid_eis_sp).detach().numpy()
        valid_x = np.stack([valid_preds_sp, valid_preds_sm]).transpose(1, 2, 0).reshape(len(valid_preds_sp), -1)
        valid_y = np.ones(len(valid_preds_sp), dtype=int)
        valid_y_hat = rfc.predict(valid_x)
        valid_df = pd.DataFrame(
            {
                "capacitance_id": np.tile(discrimination_data["capacitances"], (n_repeats,)),
                "y": valid_y,
                "y_hat": valid_y_hat,
            }
        )
        discrimination_performance[pref_c_id] = valid_df.groupby("capacitance_id").apply(
            lambda x: (x["y"] == x["y_hat"]).mean() * 100
        )
    print()
    dfrow["discrimination_performance"] = discrimination_performance
    return dfrow


models = models.apply(train_capacitance_discrimination, axis=1)

models.to_pickle("discrimination_performance.pkl")
