import itertools
import pickle
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fannypack
import numpy as np
import sdeint
import torch
from torch.utils.data import Dataset

from dynamics_learning.networks.dynamics import DynamicSystem, Pendulum, VanDerPol
from dynamics_learning.utils.data_utils import (
    GaussianSampler,
    Sampler,
    UniformSampler,
    prep_batch,
)
from dynamics_learning.utils.plot_utils import Axis, PlotSettings

# ------------------------- #
# VISUALIZATION DATACLASSES #
# ------------------------- #


@dataclass(frozen=True)
class TrajectoryNumpy:
    """Holds the information about a trajectory."""

    states: np.ndarray
    observations: Dict[str, np.ndarray]
    controls: np.ndarray


@dataclass(frozen=True)
class VisData:
    """Struct to hold data for plotting."""

    t: torch.Tensor
    y0: torch.Tensor
    y: torch.Tensor
    u: torch.Tensor
    np_t: np.ndarray
    np_y: np.ndarray
    np_u: np.ndarray
    plot_settings: PlotSettings


@dataclass(frozen=True)
class VisDataIMG(VisData):
    """Struct to hold data for plotting image data."""

    pv: torch.Tensor
    np_pv: np.ndarray


# ------- #
# HELPERS #
# ------- #


def _get_viz_data_basic(dataset: "DynamicsDataset", device: torch.device) -> VisData:
    """Helper for returning only times and associated data, no extra info."""
    assert hasattr(dataset, "_viz_data")
    assert callable(getattr(dataset, "get_default_plot_settings", None))

    #print("Contenu de dataset._viz_data :", dataset._viz_data)

    t_array = np.array([t for t, x, u in dataset._viz_data])
    x_array = np.array([x for t, x, u in dataset._viz_data])
    u_array = np.array([u for t, x, u in dataset._viz_data])

    #x_array = x_array.astype(float)

    #u_array = u_array.astype(float)


    #print("t_array:",t_array)
    #print("x_array:",x_array)
    #print("u_array:",u_array)
    
    #type_de_donnees = x_array.dtype
    #print("Type de données du tableau NumPy :", type_de_donnees)
    
    t = torch.tensor(t_array, dtype=torch.float, device=device)[0, :]
    y = torch.tensor(x_array, dtype=torch.float, device=device).transpose(0, 1)
    u = torch.tensor(u_array, dtype=torch.float, device=device).transpose(0, 1)

    #print("t:",t)
    #print("y",y)
    #print("u:",u)

    return VisData(
        t=t,
        y0=y[0],
        y=y,
        u=u,
        np_t=t.clone().cpu().numpy(),
        np_y=y.clone().cpu().numpy(),
        np_u=u.clone().cpu().numpy(),
        plot_settings=dataset.get_default_plot_settings(),
    )


# -------- #
# DATASETS #
# -------- #


@dataclass
class DatasetConfig:
    """Configuration for a Dataset for training."""

    traj_len: int
    num_viz_trajectories: int

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        raise NotImplementedError

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        raise NotImplementedError


@dataclass
class DummyDatasetConfig(DatasetConfig):
    """A dummy dataset configuration for nested definitions."""

    traj_len: int = 0
    num_viz_trajectories: int = 0


class DynamicsDataset(ABC, Dataset):
    """Abstract dataset interface for dynamic systems."""

    # TODO fix typing
    _data: Any
    # _data: List[Tuple[np.ndarray, np.ndarray]]
    _viz_data: Any
    # _viz_data: List[Tuple[np.ndarray, np.ndarray]]

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self._data)

    def __getitem__(self, idx: Union[int, torch.Tensor, List[int]]) -> torch.Tensor:
        """Get specific datapoint."""
        if torch.is_tensor(idx):
            assert isinstance(idx, torch.Tensor)  # mypy
            idx = idx.tolist()
        # TODO fix
        return self._data[idx]  # type: ignore

    @abstractmethod
    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""

    @abstractmethod
    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""

    def preprocess_data(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a batch for training."""
        _, batch_t, batch_y, batch_u = prep_batch(batch, device)
        return batch_t, batch_y, batch_u


@dataclass(frozen=True)
class ContinuousDynamicsDatasetConfig:
    """Config object for continuous dynamical systems."""

    ic: Sampler
    end_times: Sampler
    policy: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    trajectory_length: int = 20
    num_trajectories: int = 10000


class ContinuousDynamicsDataset(DynamicsDataset):
    """Dataset for DE-defined continuous dynamical systems."""

    def __init__(
        self,
        system: DynamicSystem,
        data_config: ContinuousDynamicsDatasetConfig,
        viz_config: ContinuousDynamicsDatasetConfig,
        generation_batches: int = 100,
        process_noise: Optional[np.ndarray] = None,
        measurement_var: float = 0.0,
        param_sampler: Optional[Sampler] = None,
    ) -> None:
        """Initalize a continuous system.

        Parameters
        ----------
        system : DynamicSystem
            Dynamic system.
        data_config : ContinuousDynamicsDatasetConfig
            The configuration of the data.
        viz_config : ContinuousDynamicsDatasetConfig
            The configuration of the visualization data.
        generation_batches : int
            Number of data points to generate at a time.
        process_noise : Optional[np.ndarray], default=None
            Noise to use in the SDE.
            # THIS ONLY WORKS FOR Van Der Pol and pendulum right now.
        measurement_var : float, default=0.0
            Measurement variance to be used during pre processing. By default, there is no added noise.
        """
        self._length = data_config.num_trajectories
        self._sys = system
        self._generation_batches = generation_batches
        self._process_noise = process_noise
        self._measurement_var = measurement_var
        self._param_sampler = param_sampler
        self._data = self.create_data(data_config)
        self._viz_data = self.create_data(viz_config)

    def create_data(
        self, config: ContinuousDynamicsDatasetConfig
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate a dataset from a configuration.

        Parameters
        ----------
        config : ContinuousDynamicsDatasetConfig
            The configuration of the data.

        Returns
        -------
        data : List[Tuple[np.ndarray, np.ndarray]], shape=[(T, (T, B, X))]
            List of tuples of times and batches of sequences of data.
        """
        data = []
        remaining = config.num_trajectories

        # stochastic
        if self._process_noise is not None:
            iterations = 0
            for i in range(remaining):
                tspan = np.linspace(
                    0.0, config.end_times.sample(), config.trajectory_length
                )
                x0 = config.ic.sample()

                # noise
                def G(x, t):
                    return self._process_noise

                # dynamics
                if config.policy is None:

                    def dyn(x, t):
                        return self._sys.dx(x, t, p=p, u=None)

                else:

                    def dyn(x, t):
                        return self._sys.dx(x, t, p=p, u=config.policy(x, t))

                while True:
                    # sample dynamics parameters
                    if self._param_sampler is not None:
                        p = self._param_sampler.sample()
                    else:
                        p = None

                    result = sdeint.itoint(dyn, G, x0, tspan)  # type: ignore
                    if not np.isnan(np.sum(result)):
                        # sdeint gets nan sometimes?
                        break
                    if iterations > self._generation_batches and iterations % 1000 == 0:
                        print(f"remaining: {i}")
                    iterations += 1

                # wrapping angles to [-pi, pi]
                # > https://stackoverflow.com/a/11181951
                if isinstance(self._sys, Pendulum):
                    result[:, 0] = np.arctan2(
                        np.sin(result[:, 0]), np.cos(result[:, 0])
                    )

                # reconstructing (deterministic) control inputs
                _ctrl: List[np.ndarray] = []
                for i in range(len(tspan)):
                    t = tspan[i]
                    x = result[i, :]
                    if config.policy is None:
                        _ctrl.append([])
                    else:
                        _ctrl.append(config.policy(x, t))
                ctrl: np.ndarray = np.array(_ctrl)
                data.append((tspan, result, ctrl))
            return data

        # deterministic
        while remaining > 0:
            raise NotImplementedError  # TODO: add back support for this later

            batch_size = min(remaining, self._generation_batches)
            t = np.linspace(0.0, config.end_times.sample(), config.trajectory_length)
            x0s = torch.tensor(config.ic.sample_batch(batch_size), dtype=torch.float)
            xs = self._sys.solve_torch(t, x0s)

            # wrapping angles to [-pi, pi]
            # > https://stackoverflow.com/a/11181951
            if isinstance(self._sys, Pendulum):
                xs[..., 0] = np.arctan2(np.sin(xs[..., 0]), np.cos(xs[..., 0]))

            data += [(t, xs[:, i, :]) for i in range(batch_size)]
            remaining = remaining - batch_size
        return data

    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""
        return _get_viz_data_basic(self, device)

    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""
        if isinstance(self._sys, Pendulum):
            # angular plot settings
            return PlotSettings(
                axis=Axis(xlim=(-np.pi - 0.1, np.pi + 0.1), ylim=(-7, 7),)
            )
        return PlotSettings(axis=Axis(xlim=(-5, 5), ylim=(-5, 5)))

    def preprocess_data(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See parent."""
        _, batch_t, batch_y, batch_u = prep_batch(batch, device)
        return batch_t, batch_y, batch_u


@dataclass
class OfflineDatasetConfig(DatasetConfig):
    """Dataset for training that reads from a saved pickle file."""

    paths: Tuple[str, str]
    pend_xy: bool = False

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        return (1,) #erreur si on ne met pas la bonne dimension de l'observation y

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        dataset = OfflineDataset(
            self.traj_len, self.num_viz_trajectories, self.paths[0], self.paths[1],
        )
        if self.pend_xy:
            dataset = PendulumXYDataset(dataset, has_vel=False)
        return dataset


class OfflineDataset(DynamicsDataset):
    """Dataset for DE-defined continuous dynamical systems."""

    def __init__(
        self,
        traj_len: int,
        num_viz_trajectories: int,
        train_data_path: str,
        val_data_path: str,
    ) -> None:
        """Initalize an offline saved dataset."""
        with open(train_data_path, "rb") as handle:
            self._data = pickle.load(handle)
        #print("Longueur de self._data:", len(self._data))
        #print("self._data:", self._data)
        print(len(self._data[0][0]))
        assert len(self._data[0][0]) >= traj_len
        self._data = [
            (t[:traj_len], x[:traj_len], c[:traj_len]) for (t, x, c) in self._data
        ]
        #print("self._data:" self._data)
        with open(val_data_path, "rb") as handle:
            self._viz_data = pickle.load(handle)[:num_viz_trajectories]

    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""
        return _get_viz_data_basic(self, device)

    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""
        return PlotSettings(axis=Axis(xlim=(-np.pi - 0.1, np.pi + 0.1), ylim=(-7, 7),))

    def preprocess_data(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See parent."""
        _, batch_t, batch_y, batch_u = prep_batch(batch, device)
        return batch_t, batch_y, batch_u


class DynamicsDatasetType(Enum):
    """Supported simple dynamics datasets."""

    VDP = 1
    PEND_XY = 2
    PEND_ALL = 3
    PEND_ANG = 4


@dataclass
class DynamicSystemDatasetConfig(DatasetConfig):
    """Supported synthetic datasets."""

    system: DynamicsDatasetType
    num_trajectories: int = 10000
    dt: float = 0.025
    measurement_var: float = 0.0001
    policy: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        return (2,) if self.system != DynamicsDatasetType.PEND_ALL else (4,)

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        train_end_time = (self.traj_len - 1) * self.dt
        valid_end_time = 3 * (self.traj_len - 1) * self.dt
        process_noise = np.array([[0.05, 0], [0, 0.05]])

        if self.system == DynamicsDatasetType.VDP:
            vdp_sys = VanDerPol(mu=1)
            data_config = ContinuousDynamicsDatasetConfig(
                ic=GaussianSampler(np.array([0, 0]), np.array([[4, 0], [0, 4]])),
                end_times=UniformSampler(train_end_time, train_end_time),
                policy=self.policy,
                trajectory_length=self.traj_len,
                num_trajectories=self.num_trajectories,
            )
            viz_config = ContinuousDynamicsDatasetConfig(
                ic=GaussianSampler(np.array([0, 0]), np.array([[4, 0], [0, 4]])),
                end_times=UniformSampler(valid_end_time, valid_end_time),
                policy=self.policy,
                trajectory_length=3 * (self.traj_len - 1) + 1,
                num_trajectories=self.num_viz_trajectories,
            )
            dataset = ContinuousDynamicsDataset(
                vdp_sys,
                data_config,
                viz_config,
                process_noise=process_noise,
                measurement_var=self.measurement_var,
            )

        elif (
            self.system == DynamicsDatasetType.PEND_ANG
            or self.system == DynamicsDatasetType.PEND_XY
            or self.system == DynamicsDatasetType.PEND_ALL
        ):
            p_sys = Pendulum()

            # angular dataset
            data_config = ContinuousDynamicsDatasetConfig(
                ic=UniformSampler(np.array([-np.pi, -2]), np.array([np.pi, 2])),
                end_times=UniformSampler(train_end_time, train_end_time),
                policy=self.policy,
                trajectory_length=self.traj_len,
                num_trajectories=self.num_trajectories,
            )
            viz_config = ContinuousDynamicsDatasetConfig(
                ic=UniformSampler(np.array([-np.pi, -2]), np.array([np.pi, 2])),
                end_times=UniformSampler(valid_end_time, valid_end_time),
                policy=self.policy,
                trajectory_length=3 * (self.traj_len - 1) + 1,
                num_trajectories=self.num_viz_trajectories,
            )
            dataset = ContinuousDynamicsDataset(
                p_sys,
                data_config,
                viz_config,
                process_noise=process_noise,
                measurement_var=self.measurement_var,
            )

            # XY or image dataset
            if (
                self.system == DynamicsDatasetType.PEND_XY
                or self.system == DynamicsDatasetType.PEND_ALL
            ):
                dataset = PendulumXYDataset(
                    dataset, has_vel=self.system == DynamicsDatasetType.PEND_ALL,
                )
        return dataset



