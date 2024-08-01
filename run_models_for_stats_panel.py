import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from collections import OrderedDict
import sys
import copy

sys.path.append("../efish-physics-model/objects")
sys.path.append("../efish-physics-model/helper_functions")
sys.path.append("../efish-physics-model/uniform_points_generation")
sys.path.append("./end-to-end")
sys.path.append("./end-to-end-with-feedback")
sys.path.append("./electric-properties-only")

from electric_images_dataset import ElectricImagesDataset  # noqa: 402
from EndToEndConvNN_PL import EndToEndConvNN_PL  # noqa: 402
from EndToEndConvNNWithFeedback_PL import EndToEndConvNNWithFeedback_PL  # noqa: 402

# seed everything
gpus = [3]
random_seed = 113
L.seed_everything(random_seed)

# HYPERPARAMETERS OF THE RUN
rc_lambdas = [0, 1, 2, 4, 8, 16]
data_dir_name = "../efish-physics-model/data/processed/data-2024_06_18-characterization_dataset"
batch_size = 5000
max_epochs = 10
input_noise_std = 0.01
input_noise_type = "additive"
activation = "relu"
activation_spatial = "relu"
activation_feedback = "relu"
model_type = "two_paths"
poly_degree_radius = 3
poly_degree_distance = 4
average_pooling_kernel_size = 7


# PARAMETERS FOR ALL MODELS
in_ch = 1  # corresponds to TwoPaths model

dset = ElectricImagesDataset(data_dir_name=data_dir_name, fish_t=20, fish_u=30)
dloader = DataLoader(dset, batch_size=4, shuffle=True)
# data loaders
train_dset, valid_dset = torch.utils.data.random_split(dset, [0.75, 0.25])  # type: ignore
train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12)
valid_loader = DataLoader(valid_dset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=12)

layers_properties = OrderedDict(
    [
        (
            "conv1",
            dict(in_channels=in_ch, out_channels=8, kernel_size=7, stride=1, max_pool=dict(kernel_size=3, stride=1)),
        ),
        (
            "conv2",
            dict(in_channels=8, out_channels=8, kernel_size=5, stride=1),
        ),
        (
            "conv3",
            dict(in_channels=8, out_channels=8, kernel_size=5, stride=1, max_pool=dict(kernel_size=3, stride=2)),
        ),
        # the fully connected layers can have dropout or flatten layers - some can miss the activation
        ("fc1", dict(dropout=0.5, flatten=True, in_features=None, out_features=960)),
        ("fc2", dict(dropout=0.5, in_features=960, out_features=240)),
        ("fc3", dict(in_features=240, out_features=4, activation=False)),
    ]
)

layers_properties_full = copy.deepcopy(layers_properties)
layers_properties_full["fc3"]["out_features"] = 6

for lambda_rc in rc_lambdas:
    ####################################################################################
    ####################################################################################
    # feedback with VALUES
    ####################################################################################
    ####################################################################################

    use_estimates_as_feedback = False

    model_PL = EndToEndConvNNWithFeedback_PL(
        # spatial model properties
        layers_properties=copy.deepcopy(layers_properties),
        activation_spatial=activation_spatial,
        model_type=model_type,
        # feedback model properties (for extracting electric properties)
        kernel_size=average_pooling_kernel_size,
        in_channels=2,
        poly_degree_distance=poly_degree_distance,
        poly_degree_radius=poly_degree_radius,
        activation_feedback=activation_feedback,
        # miscellaneous properties
        use_estimates_as_feedback=use_estimates_as_feedback,
        input_noise_std=input_noise_std,
        input_noise_type=input_noise_type,
        loss_lambda=[1, 1, 1, 1, lambda_rc, lambda_rc],
    )

    # dummy forward pass to initialize the model
    batch = next(iter(dloader))
    _ = model_PL.model(
        batch[0],
        distances=torch.zeros(batch[0].shape[0]).to(batch[0].device),
        radii=torch.zeros(batch[0].shape[0]).to(batch[0].device),
    )

    logger = pl_loggers.TensorBoardLogger(
        save_dir=f"./stats-panel/feedback-with-values-randseed_{random_seed}-lambdaRC_{lambda_rc}/"
    )
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger, devices=gpus)
    trainer.fit(model=model_PL, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    ####################################################################################
    ####################################################################################
    # full models
    ####################################################################################
    ####################################################################################

    model_PL = EndToEndConvNN_PL(
        layers_properties=copy.deepcopy(layers_properties_full),
        activation=activation,
        input_noise_std=input_noise_std,
        input_noise_type=input_noise_type,
        model_type=model_type,
        loss_lambda=[1, 1, 1, 1, lambda_rc, lambda_rc],
    )

    batch = next(iter(dloader))
    _ = model_PL.model(batch[0])

    logger = pl_loggers.TensorBoardLogger(
        save_dir=f"./stats-panel/full-model-randseed_{random_seed}-lambdaRC_{lambda_rc}/"
    )
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger, devices=gpus)
    trainer.fit(model=model_PL, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    ####################################################################################
    ####################################################################################
    # feedback with ESTIMATES
    ####################################################################################
    ####################################################################################

    use_estimates_as_feedback = True

    model_PL = EndToEndConvNNWithFeedback_PL(
        # spatial model properties
        layers_properties=copy.deepcopy(layers_properties),
        activation_spatial=activation_spatial,
        model_type=model_type,
        # feedback model properties (for extracting electric properties)
        kernel_size=average_pooling_kernel_size,
        in_channels=2,
        poly_degree_distance=poly_degree_distance,
        poly_degree_radius=poly_degree_radius,
        activation_feedback=activation_feedback,
        # miscellaneous properties
        use_estimates_as_feedback=use_estimates_as_feedback,
        input_noise_std=input_noise_std,
        input_noise_type=input_noise_type,
        loss_lambda=[1, 1, 1, 1, lambda_rc, lambda_rc],
    )

    # dummy forward pass to initialize the model
    batch = next(iter(dloader))
    _ = model_PL.model(batch[0])

    logger = pl_loggers.TensorBoardLogger(
        save_dir=f"./stats-panel/feedback-with-estimates-randseed_{random_seed}-lambdaRC_{lambda_rc}/"
    )
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger, devices=gpus)
    trainer.fit(model=model_PL, train_dataloaders=train_loader, val_dataloaders=valid_loader)
