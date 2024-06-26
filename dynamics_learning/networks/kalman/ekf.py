from dataclasses import dataclass
from typing import List, Optional, Tuple, overload

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint as odeint

from dynamics_learning.networks.dynamics import LinearSystem
from dynamics_learning.networks.kalman.core import (
    KalmanEstimator,
    KalmanEstimatorCell,
    KalmanEstimatorConfig,
)
from dynamics_learning.networks.models import MLPConfig, SimpleODENetConfig
from dynamics_learning.utils.net_utils import batch_eye, reg_psd, reparameterize_gauss


@dataclass
class EKSmoothCache:
    """Cache to store values needed for Extended Kalman smoothing.

    On forward pass, we cache the means and covariances as well as a smoothing param.
    For discrete smoothing, this is the Jacobian. For continuous-discrete smoothing,
    this is the smoothing gain.

    "minus" denotes pre-measurement update.

    Source:
    https://users.aalto.fi/~ssarkka/pub/cdgen-smooth.pdf
    """

    mu_tk: List[torch.Tensor]
    cov_tk: List[torch.Tensor]
    mu_tk_minus: List[torch.Tensor]
    cov_tk_minus: List[torch.Tensor]
    G_tk: List[torch.Tensor]
    Cs: List[torch.Tensor]  # only used for testing and debugging

    @staticmethod
    def create_empty() -> "EKSmoothCache":
        """Create empty cache."""
        return EKSmoothCache([], [], [], [], [], [])

    @property
    def T(self) -> int:
        """Cache size."""
        return len(self.mu_tk)


class EKFCell(KalmanEstimatorCell):
    """An EKF cell."""

    def __init__(
        self,
        dynamics: nn.Module,
        observation_dynamics: Optional[nn.Module] = None,
        latent_dim: int = 2,
        observation_dim: int = 1,
        ctrl_dim: int = 0,
        initial_state: Optional[np.ndarray] = None,
        initial_variance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
        is_continuous: bool = True,
        const_var: bool = False,
        reparam: bool = False,
        regularizer: float = 1e-3,
    ) -> None:
        """See parent class."""
        super(EKFCell, self).__init__(
            dynamics,
            observation_dynamics=observation_dynamics,
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            ctrl_dim=ctrl_dim,
            initial_state=initial_state,
            initial_variance=initial_variance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            is_continuous=is_continuous,
            const_var=const_var,
            reparam=reparam,
            regularizer=regularizer,
        )
        self._smooth_cache = EKSmoothCache.create_empty()

    # --------- #
    # UTILITIES #
    # --------- #

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._smooth_cache = EKSmoothCache.create_empty()

    def get_initial_hidden_state(
        self, batch_size: int, z0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return a nominal initial hidden state.

        Parameters
        ----------
        batch_size : int
            Batch size
        z0 : torch.tensor, shape=(B, n)
            Batch of initial latent states.

        Returns
        -------
        torch.Tensor
            Batched initial hidden state
        """
        eye = batch_eye(self._latent_dim, batch_size, device=self._device)

        if z0 is None:
            z0 = eye @ self._z0

        cov0 = eye @ self.P0
        return self.gaussian_parameters_to_vector(z0, cov0)

    @overload
    def vector_to_gaussian_parameters(
        self, zp_vect: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See vector_to_gaussian_parameters below."""

    @overload
    def vector_to_gaussian_parameters(
        self, zp_vect: torch.Tensor, return_Cs: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See vector_to_gaussian_parameters below."""

    def vector_to_gaussian_parameters(
        self, zp_vect: torch.Tensor, return_Cs: bool = False
    ):
        """Take a packed vector and convert it into mean and cov.

        Parameters
        ----------
        zp_vect : torch.Tensor, shape=(B, n + n**2) or (B, n + 2 * n**2)
            The entire vectorized latent belief. n is the latent dimension.
         return_Cs : bool
            Flag for returning intermediate smoothing dynamics variable. See:
            https://users.aalto.fi/~ssarkka/pub/cdgen-smooth.pdf
            Equation (38)

        Returns
        -------
        mu : torch.Tensor, shape=(B, n)
            Mean of the state.
        cov : torch.Tensor, shape=(B, n, n)
            Variance of the state.
        """
        c = zp_vect.shape[-1]

        if self._is_continuous:
            assert np.sqrt(8 * c + 1) % 1 == 0
            assert (np.sqrt(8 * c + 1) - 1) % 4 == 0

            z_dim = int(round((np.sqrt(8 * c + 1) - 1) / 4))

            mu = zp_vect[..., :z_dim]
            cov = zp_vect[..., z_dim : (z_dim + z_dim ** 2)]
            Cs = zp_vect[..., (z_dim + z_dim ** 2) :]

            cov_shape = cov.shape[:-1] + (z_dim, z_dim)
            Cs_shape = Cs.shape[:-1] + (z_dim, z_dim)
            cov = cov.reshape(cov_shape)
            cov = reg_psd(0.5 * (cov + cov.transpose(-1, -2)), reg=self._reg)
            Cs = Cs.reshape(Cs_shape)

            if return_Cs:
                return mu, cov, Cs
            else:
                return mu, cov

        else:
            assert return_Cs is False
            assert np.sqrt(4 * c + 1) % 1 == 0
            assert (np.sqrt(4 * c + 1) - 1) % 2 == 0

            z_dim = int(round((np.sqrt(4 * c + 1) - 1) / 2))

            mu = zp_vect[..., :z_dim]
            cov = zp_vect[..., z_dim:]

            cov_shape = cov.shape[:-1] + (z_dim, z_dim)
            cov = cov.reshape(cov_shape)
            cov = reg_psd(0.5 * (cov + cov.transpose(-1, -2)), reg=self._reg)

            return mu, cov

    def gaussian_parameters_to_vector(
        self, mu: torch.Tensor, cov: torch.Tensor, Cs=None, dcov=None,
    ) -> torch.Tensor:
        """
        Take the latent mean and covariance and convert it to a vector.

        Parameters
        ----------
        mu : torch.Tensor, shape=(B, n)
            Mean of the latent state.
        cov : torch.Tensor, shape=(B, n, n)
            Variance of the latent state.
        dcov : bool
            Derivative of the covariance. If this is provided, the derivative of
            covariance will be returned in the vector.

        Returns
        -------
        torch.Tensor
            - shape=(B, n + n**2), or
            - [DEP] shape=(B, n + n*(n+1)/2) if use_cholesky == True, or
            - shape=(B, n + 2*n**2) if use_smooth == True
            The entire state, vectorized.
        """
        tensor_shape = mu.shape[:-1]
        z_dim = mu.shape[-1]

        if dcov is not None:
            cov_flat = dcov.reshape(*tensor_shape, z_dim ** 2)
        else:
            cov_flat = cov.reshape(*tensor_shape, z_dim ** 2)

        if self._is_continuous:
            if Cs is None:
                Cs = cov
            Cs_flat = Cs.reshape(*tensor_shape, z_dim ** 2)

            return torch.cat((mu, cov_flat, Cs_flat), dim=-1)
        else:
            assert dcov is None
            assert Cs is None

            return torch.cat((mu, cov_flat), dim=-1)

    def latent_to_observation(
        self,
        z_mu: torch.Tensor,
        z_cov: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See parent class."""
        if self._reparam:
            z_mu = reparameterize_gauss(z_mu, z_cov, device=self._device)
            y_mu = self.observation_dynamics(z_mu, cond=cond)
            shape = y_mu.shape
            return y_mu, self.R.repeat(list(shape[:-1]) + [1, 1])
        else:
            y_mu = self.observation_dynamics(z_mu, cond=cond)
            shape = y_mu.shape
            Z = z_mu.shape[-1]
            Y = y_mu.shape[-1]
            if cond is not None:
                cond = cond.reshape(-1, cond.shape[-1])
            C = self.get_jacobian(
                z_mu.new_tensor(0), z_mu.reshape(-1, Z), cond=cond, mode="observation",
            ).reshape(list(shape[:-1]) + [Y, Z])
            y_cov = C @ z_cov @ C.transpose(-1, -2) + self.R.repeat(
                list(shape[:-1]) + [1, 1]
            )
            return y_mu, y_cov

    @torch.enable_grad()
    def get_jacobian(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        mode: str = "dynamics",
    ) -> torch.Tensor:
        """
        Compute the jacobian of a function.

        Parameters
        ----------
        t : torch.Tensor, shape=(1)
            The current time.
        z : torch.Tensor, shape=(B, n)
            The current latent state.
        u : Optional[torch.Tensor], shape=(B, m), default=None
            The control input.
        cond : Optional[torch.Tensor], shape=(B, C), default=None
            Conditional context.
        mode : str, default="dynamics"
            Flag indicating the type of Jacobian. Choices: {"dynamics", "observation"}.

        Returns
        -------
        torch.Tensor, shape=(B, n, n)
            The jacobian with respect to the latent state.
        """
        # shapes
        B, n = z.shape
        p = self._observation_dim
        out_dim = n if mode == "dynamics" else p

        # batched jacobian trick
        # > https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
        # > comment from MasanoriYamada
        z_jac = z.unsqueeze(1).repeat(1, out_dim, 1)  # (B, out_dim, n)
        z_jac.requires_grad_(True)

        # reshaping conditional context for jacobian computation
        if cond is not None and cond.shape[-1] > 0:
            C = cond.shape[-1]
            c_jac = cond.unsqueeze(1).repeat(1, out_dim, 1)  # (B, n, C)
            c_jac = c_jac.reshape(-1, C)
        else:
            c_jac = torch.zeros(B * out_dim, 0, device=self._device)

        if mode == "dynamics":
            if isinstance(self._dynamics, LinearSystem):
                assert cond is None  # TODO: fix this later
                return self._dynamics.A.expand([B, n, n])

            # reshaping control input for jacobian computation
            if u is not None and u.shape[-1] > 0:
                m = u.shape[-1]
                u_jac = u.unsqueeze(1).repeat(1, n, 1)  # (B, n, m)
                u_jac = u_jac.reshape(-1, m)
            else:
                u_jac = torch.zeros(B * n, 0, device=self._device)

            y = self._dynamics(t, z_jac.reshape(-1, n), u_jac, cond=c_jac).reshape(
                B, -1, n
            )

        elif mode == "observation":
            if isinstance(self._observation_dynamics, LinearSystem):
                assert cond is None  # TODO: fix this later
                return self._observation_dynamics.A.expand([B, p, n])  # ?

            y = self._observation_dynamics(z_jac.reshape(-1, n), cond=c_jac).reshape(
                B, -1, p
            )

        else:
            raise NotImplementedError

        # compute jacobian
        mask = batch_eye(out_dim, B).to(z.device)
        jac = torch.autograd.grad(y, z_jac, mask, create_graph=True)
        return jac[0]

    # --------------- #
    # FILTER/SMOOTHER #
    # --------------- #

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """See parent class."""
        if self._is_continuous:
            mu, cov, Cs = self.vector_to_gaussian_parameters(z, return_Cs=True)
            A = self.get_jacobian(t, mu, u=u, cond=cond)
            A_T = A.transpose(-1, -2)

            dcov = A @ cov + cov @ A_T + self.Q

            cov = 0.5 * (cov + cov.transpose(-1, -2))
            dcov = 0.5 * (dcov + dcov.transpose(-1, -2))

            mu_next = self._dynamics(t, mu, u, cond=cond)
            c_dot = Cs @ A_T
            dynamics = self.gaussian_parameters_to_vector(
                mu_next, cov, Cs=c_dot, dcov=dcov
            )

        else:
            mu, cov = self.vector_to_gaussian_parameters(z)

            A = self.get_jacobian(t, mu, u=u, cond=cond)
            A_T = A.transpose(-1, -2)

            cov_next = A @ cov @ A_T + self.Q
            cov_next = 0.5 * (cov_next + cov_next.transpose(-1, -2))

            self._smooth_cache.G_tk.append(A)

            mu_next = self._dynamics(t, mu, u, cond=cond)
            dynamics = self.gaussian_parameters_to_vector(mu_next, cov_next, dcov=None)

        return dynamics

    def measurement_update(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """See parent class."""
        # unpack latent belief. Cs is a smoothing parameter.
        if self._is_continuous:
            mu_p, cov_p, Cs = self.vector_to_gaussian_parameters(z, return_Cs=True)
            self._smooth_cache.G_tk.append(Cs @ torch.inverse(cov_p))
            self._smooth_cache.Cs.append(Cs)
        else:
            mu_p, cov_p = self.vector_to_gaussian_parameters(z)

        y_p = self.observation_dynamics(mu_p, cond=cond)  # observe on predicted mean
        
        #print("Value of mu_p:", mu_p)
        #print("Value of y_p:", y_p)

        # observation Jacobian. NOT the same as Cs above.
        C = self.get_jacobian(t, mu_p, mode="observation", cond=cond)
        C_T = C.transpose(-1, -2)
        K = cov_p @ C_T @ torch.inverse(C @ cov_p @ C_T + self.R)  # Kalman gain

        # measurement updates. Joseph form for covariance update.
        mu_new = mu_p + (K @ (y - y_p).unsqueeze(-1)).squeeze(-1)
        J = torch.eye(self._latent_dim, device=self._device) - K @ C
        cov_new = J @ cov_p @ J.transpose(-1, -2) + K @ self.R @ K.transpose(-1, -2)
        cov_new = reg_psd(0.5 * (cov_new + cov_new.transpose(-1, -2)), reg=self._reg)

        # caching for smoothing
        self._smooth_cache.mu_tk_minus.append(mu_p)
        self._smooth_cache.cov_tk_minus.append(cov_p)
        self._smooth_cache.mu_tk.append(mu_new)
        self._smooth_cache.cov_tk.append(cov_new)

        return self.gaussian_parameters_to_vector(mu_new, cov_new)

    def smooth(self) -> torch.Tensor:
        """RTS smoother.

        Executes Rauch-Tung-Striebel (forward-backward) smoothing on the
        filtered data using the cached values.

        Returns
        -------
        zs_vect : torch.Tensor, shape=(B, n + n ** 2)
            A batch of packed vectors of smoothed belief distributions.
        """
        mu_smooth = self._smooth_cache.mu_tk
        cov_smooth = self._smooth_cache.cov_tk

        for t in reversed(range(self._smooth_cache.T - 1)):
            mu_p = self._smooth_cache.mu_tk_minus[t + 1]
            cov_p = self._smooth_cache.cov_tk_minus[t + 1]

            if self._is_continuous:
                # continuous: retrieve cached gain directly
                Kt = self._smooth_cache.G_tk[t + 1]
            else:
                # discrete: retrieve cached Jacobian
                A_t = self._smooth_cache.G_tk[t]
                Kt = cov_smooth[t] @ A_t.transpose(-1, -2) @ torch.inverse(cov_p)

            mu_smooth[t] = mu_smooth[t] + (
                Kt @ (mu_smooth[t + 1] - mu_p).unsqueeze(-1)
            ).squeeze(-1)
            cov_smooth[t] = cov_smooth[t] + Kt @ (
                cov_smooth[t + 1] - cov_p
            ) @ Kt.transpose(-1, -2)

        mu_smooth_tensor = torch.stack(mu_smooth)
        cov_smooth_tensor = torch.stack(cov_smooth)
        cov_smooth_tensor = reg_psd(
            0.5 * (cov_smooth_tensor + cov_smooth_tensor.transpose(-1, -2)),
            reg=self._reg,
        )  # DEBUG - see other comment

        return self.gaussian_parameters_to_vector(mu_smooth_tensor, cov_smooth_tensor)


class EKFEstimator(KalmanEstimator):
    """Estimator for EKF cell."""

    def __init__(self, config: "EKFEstimatorConfig", cell: EKFCell) -> None:
        """Initialize an EKF estimator."""
        super(EKFEstimator, self).__init__(config)

        self._cell = cell
        self._is_continuous = config.is_continuous

    @property
    def cell(self) -> EKFCell:
        """EKF Cell."""
        return self._cell

    def _hidden_vector_to_obs(
        self,
        vectorized_hidden_states: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts a belief distributions over hidden states to distributions over obs."""
        z_mu, z_cov = self.cell.vector_to_gaussian_parameters(vectorized_hidden_states)
        if return_hidden:
            return z_mu, z_cov
        else:
            return self.latent_to_observation(z_mu, z_cov, cond=cond)

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        u: torch.Tensor,
        z0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a filtering forward pass. Optionally, also smooth the outputs.

        NOTE: the indexing convention is extremely atypical. The initial belief
        distribution is a PRE-measurement update belief. In other words,
        we specify a prior over mu_{0|1} and cov_{0|1}, whereas the
        standard specification is a belief over mu_{1|1} and cov_{1|1}.

        Parameters
        ----------
        t : torch.Tensor, shape=(T)
            Current time.
        y : torch.Tensor, shape=(T, B, p)
            Batch of observation trajectories.
        u : torch.Tensor, shape=(T, B, m), default=None
            Batch of control input trajectories. We assume a ZOH input signal at the
            frequency implied by the times provided in t.
        z0 : torch.Tensor, shape=(B, n)
            Batch of initial hidden states.
        cond : Optional[torch.Tensor], shape=(B, C)
            Conditional context.
        return_hidden : bool, default=False
            Flag indicating whether to return hidden states instead of obs.

        Returns
        -------
        (z_mu, z_cov) if return_hidden=True, shape=((T, B, n), (T, B, n, n))
            Batches of trajectories of latent distributions.

        OR

        (y_mu, y_cov) if return_hidden=False, shape=((T, B, n), (T, B, n, n))
            Batches of trajectories of observation distributions.
        """
        self.cell.clear_cache()

        z_t = []  # stores posterior latent distributions
        z_t_minus = [z0]
        z_next = z0

        # filtering
        if self._is_continuous:
            for t_i, t_i_1, y_i, u_i in zip(t[:-1], t[1:], y[:-1], u[:-1]):
                z_next = self.cell.measurement_update(t_i, y_i, z_next, cond=cond)
                z_t.append(z_next)
                z_next = odeint(
                    lambda t, z: self.cell(t, z, u_i, cond=cond),
                    z_next,
                    torch.tensor(
                        [t_i, t_i_1], dtype=torch.float, device=self.cell._device
                    ),
                    rtol=self._rtol,
                    atol=self._atol,
                )[1, :, :]
                z_t_minus.append(z_next)
        else:
            for t_i, y_i, u_i in zip(t[:-1], y[:-1], u[:-1]):
                z_next = self.cell.measurement_update(t_i, y_i, z_next, cond=cond)
                z_t.append(z_next)
                z_next = self.cell(t_i, z_next, u_i, cond=cond)
                z_t_minus.append(z_next)

        #print("t:",t)
        #print("y:",y)
        z_next = self.cell.measurement_update(t[-1], y[-1], z_next, cond=cond)
        z_t.append(z_next)
        vectorized_hidden_states = torch.stack(z_t)

        return self._hidden_vector_to_obs(
            vectorized_hidden_states, cond=cond, return_hidden=return_hidden,
        )

    def get_smooth(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return smoothed distributions."""
        assert self._is_smooth
        vectorized_z_smooth = self.cell.smooth()
        z_mu_s, z_cov_s = self.cell.vector_to_gaussian_parameters(vectorized_z_smooth)
        return z_mu_s, z_cov_s

    def predict(
        self,
        z0_mu: torch.Tensor,
        z0_cov: torch.Tensor,
        pred_times: torch.Tensor,
        u: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See parent class."""
        z0 = self.cell.gaussian_parameters_to_vector(z0_mu, z0_cov)

        # propagating latent dynamics
        z_next = z0
        z_list = [z_next]

        if self._is_continuous:
            for t_i, t_i_1, u_i in zip(pred_times[:-1], pred_times[1:], u[:-1]):
                z_next = odeint(
                    lambda t, z: self.cell(t, z, u_i, cond=cond),
                    z_next,
                    torch.tensor(
                        [t_i, t_i_1], dtype=torch.float, device=self.cell._device
                    ),
                    rtol=self._rtol,
                    atol=self._atol,
                )[1, :, :]
                z_list.append(z_next)
        else:
            for t_i, u_i in zip(pred_times[:-1], u[:-1]):
                z_next = self.cell(t_i, z_next, u_i, cond=cond)
                z_list.append(z_next)

        vectorized_hidden_states = torch.stack(z_list)
        return self._hidden_vector_to_obs(
            vectorized_hidden_states, cond=cond, return_hidden=return_hidden,
        )


@dataclass(frozen=True)
class EKFEstimatorConfig(KalmanEstimatorConfig):
    """EKF specific configuration parameters."""

    def create(self) -> EKFEstimator:
        """Create EKF from configuration."""
        dynamics = SimpleODENetConfig(
            input_dim=self.latent_dim + self.ctrl_dim + self.cond_dim,
            output_dim=self.latent_dim,
            hidden_layers=self.dyn_layers,
            hidden_units=self.dyn_hidden_units,
            nonlinearity=self.dyn_nonlinearity,
            skip=not self.is_continuous,
        )
        obs = MLPConfig(
            input_dim=self.latent_dim + self.cond_dim,
            output_dim=self.dataset.obs_shape[0],
            hidden_layer_sizes=self.obs_layers * [self.obs_hidden_units],
            nonlinearity=self.obs_nonlinearity,
        )
        cell = EKFCell(
            dynamics=dynamics.create(),
            observation_dynamics=obs.create(),
            latent_dim=self.latent_dim,
            observation_dim=self.dataset.obs_shape[0],
            is_continuous=self.is_continuous,
        )
        return EKFEstimator(self, cell)
