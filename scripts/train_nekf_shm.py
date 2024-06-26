from collections import namedtuple

from fannypack.utils import pdb_safety_net
from torch import nn as nn

from dynamics_learning.data.datasets import (
    OfflineDatasetConfig,
)
from dynamics_learning.networks.kalman.core import KalmanEstimatorConfig
from dynamics_learning.networks.estimator import EstimatorConfig
from dynamics_learning.networks.kalman.ekf import EKFEstimatorConfig
from dynamics_learning.training.configs import ExpConfig
from dynamics_learning.training.experiments import train
from dynamics_learning.custom.lr_functions import lr1

model_config: KalmanEstimatorConfig
exp_config: ExpConfig
pdb_safety_net()


# Configure hyper parameters
hyperparameter_defaults = dict(batch_size=19, learning_rate=1e-3, epochs=6000000, latent_dim=3) #nb iter = (nombre_de_trajectoires)/batch_size)*epochs

HyperParameterConfig = namedtuple(
    "HyperParameterConfig", list(hyperparameter_defaults.keys())
)
hy_config = HyperParameterConfig(**hyperparameter_defaults)

dataset_config = OfflineDatasetConfig(
    traj_len=20000,
    num_viz_trajectories=1,
    paths=['./converted_data/SHM_train_1channel_6.pickle','./converted_data/SHM_val_1channel_6.pickle'],
)

# EKF settings
model_config = EKFEstimatorConfig(
    is_smooth=True,
    latent_dim=hy_config.latent_dim,
    ctrl_dim=1,
    dataset=dataset_config, 
    dyn_hidden_units=128,
    dyn_layers=3,
    dyn_nonlinearity=nn.Softplus(beta=2, threshold=20),
    obs_hidden_units=128,
    obs_layers=3,
    obs_nonlinearity=nn.Softplus(beta=2, threshold=20),
    ramp_iters=100, # à changer car filter_length croit trop lentement pour un nombre de point trop grand
    burn_in=100,
    dkl_anneal_iter=1000,
    alpha=0.5,
    beta=1.0,
    atol=1e-9,  # default: 1e-9
    rtol=1e-7,  # default: 1e-7
    z_pred=False,
)

'''def lr1(step: int, base_lr: float) -> float:
    lr = base_lr
    _lr = lr * 0.975 ** (step // 100)
    return max(_lr, lr * 1e-2)  # default'''

# experiment settings
exp_config = ExpConfig(
    name="nekf_shm_1channel_50",
    model=model_config,
    ramp_iters=model_config.ramp_iters,
    batch_size=hy_config.batch_size,
    epochs=hy_config.epochs,
    log_iterations_simple=5,
    log_iterations_images=5,
    base_learning_rate=hy_config.learning_rate,
    learning_rate_function=lr1,
    gradient_clip_max_norm=500,
)
train(exp_config)  # train the model
