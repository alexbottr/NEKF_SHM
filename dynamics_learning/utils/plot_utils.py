from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")


DEFAULT_STYLE_LIST = ("r-", "b-", "y-", "o-", "p-")
DEFAULT_STARTMARK_LIST = ("ro", "bo", "yo", "oo", "po")
DEFAULT_ENDMARK_LIST = ("rx", "bx", "yx", "ox", "px")


@dataclass
class Axis:
    """Settings for a Plot Handler."""

    xlim: Tuple[float, float] = (-5.0, 5.0)
    ylim: Tuple[float, float] = (-5.0, 5.0)


@dataclass
class PlotSettings:
    """Settings for a Plot Handler."""

    axis: Optional[Axis] = None


@dataclass
class _PlotHandler:
    """Docstring for PlotHandler."""

    @contextmanager
    def plot_context(
        self,
        settings: PlotSettings = PlotSettings(),
        sp_shape: Optional[Tuple[int, int]] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Iterator:
        """Initialize a 2D plot within a context."""
        if sp_shape is not None:
            # if custom shape is passed, assume plot customized locally
            if size is not None:
                fig, ax = plt.subplots(*sp_shape, figsize=size)
            else:
                fig, ax = plt.subplots(*sp_shape)
        else:
            if size is not None:
                fig, ax = plt.subplots(figsize=size)
            else:
                fig, ax = plt.subplots()
            if settings.axis is not None:
                assert settings.axis is not None
                ax.set_xlim(settings.axis.xlim)
                ax.set_ylim(settings.axis.ylim)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
        try:
            yield fig, ax
        finally:
            plt.close(fig)

    @contextmanager
    def plot3d_context(self, settings: PlotSettings = PlotSettings()) -> Iterator:
        """Initialize a 3D plot within a context."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        try:
            yield fig, ax
        finally:
            plt.close(fig)

    def plot_xy_compare(
        self,
        data_list: List[np.ndarray],
        style_list: Sequence[Optional[str]] = DEFAULT_STYLE_LIST,
        startmark_list: Sequence[Optional[str]] = DEFAULT_STARTMARK_LIST,
        endmark_list: Sequence[Optional[str]] = DEFAULT_ENDMARK_LIST,
        xlabel: str = "x",
        ylabel: str = "y",
    ) -> None:
        """Compare multiple sets of data on one plot."""
        for idx, data in enumerate(data_list):
            self.plot_xy(
                data,
                style=style_list[idx],
                startmark=startmark_list[idx],
                endmark=endmark_list[idx],
                xlabel=xlabel,
                ylabel=ylabel,
            )

    def plot_timeseries_compare(
        self,
        time_list: List[np.ndarray],
        data_list: List[np.ndarray],
        ylabel: str,
        style_list: Sequence[Optional[str]] = DEFAULT_STYLE_LIST,
        startmark_list: Sequence[Optional[str]] = DEFAULT_STARTMARK_LIST,
        endmark_list: Sequence[Optional[str]] = DEFAULT_ENDMARK_LIST,
        xlabel: str = "t",
    ) -> None:
        """Compare time series data against each other."""
        for idx, data in enumerate(data_list):
            self.plot_timeseries(
                time_list[idx],
                data_list[idx],
                ylabel,
                style=style_list[idx],
                startmark=startmark_list[idx],
                endmark=endmark_list[idx],
                xlabel=xlabel,
            )

        # limits
        t_min = np.min(np.concatenate(time_list))
        t_max = np.max(np.concatenate(time_list))
        #print("data_list:", data_list)
        #y_min = np.min(np.concatenate(data_list)) #on ajuste les ymin et ymax pour mieux voir la courbe
        #y_max = np.max(np.concatenate(data_list))
        #print(data_list)
        #print(data_list[1])
        #y_min = 1.3*np.min(np.concatenate(data_list[1])) #on ajuste les ymin et ymax pour mieux voir la courbe
        #y_max = 1.3*np.max(np.concatenate(data_list[1]))
        t_range = t_max - t_min
        plt.xlim((t_min - 0.02 * t_range, t_max + 0.02 * t_range))
        plt.ylim((-2, 2))
        #plt.ylim((y_min, y_max))

    def plot_xy(
        self,
        data: np.ndarray,
        style: Optional[str] = None,
        startmark: Optional[str] = None,
        endmark: Optional[str] = None,
        xlabel: str = "x",
        ylabel: str = "y",
    ) -> None:
        """Plot XY data.

        Parameters
        ----------
        data : torch.Tensor, shape=(L, B, 2)
            Input data to be plotted.
        style : str
            Style of the plotted data.
        startmark : bool
            Flag for starting marker.
        endmark : bool
            Flag for end marker.
        xlabel : str
            Custom x label.
        ylabel : str
            Custom y label.
        """
        x = data[:, :, 0]
        y = data[:, :, 1]

        # starting/ending markers for trajectories
        if startmark is not None:
            plt.plot(x[0, :], y[0, :], startmark)  # initial condition
        if endmark is not None:
            plt.plot(x[-1, :], y[-1, :], endmark, markersize=8)  # final state

        # linestyle
        if style is not None:
            plt.plot(x, y, style)
        else:
            plt.plot(x, y)

        # labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def plot_timeseries(
        self,
        time: np.ndarray,
        data: np.ndarray,
        ylabel: str,
        style: Optional[str] = None,
        startmark: Optional[str] = None,
        endmark: Optional[str] = None,
        xlabel: Optional[str] = "t",
    ) -> None:
        """Plot time series data."""
        B = data.shape[1]  # batch size

        # start/end marks
        if startmark is not None:
            plt.plot(time[0] * np.ones(B), data[0, ...], startmark)
        if endmark is not None:
            plt.plot(time[-1] * np.ones(B), data[-1, ...], endmark)

        # linestyle
        if style is None:
            plt.plot(time, data)
        else:
            plt.plot(time, data, style)

        # labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def plot_traj(self, time: np.ndarray, trajs: np.ndarray, **kwargs: Any) -> None:
        """Plot trajectories.

        Parameters
        ----------
        time : np.ndarray
            Time associated with the trajectory/trajectories.
        trajs : np.ndarray
            Trajectories to be plotted. Can be a single trajectory or list of trajectories.
        kwargs : Any
            Arguments to be passed into the plot function. Examples:
              - color, label, linestyle ...
            See the matplot lib documentation for more details
            https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html.
        """
        plt.plot(time, trajs, **kwargs)

    def plot_traj_with_var(
        self,
        time: np.ndarray,
        trajs: np.ndarray,
        var: np.ndarray,
        sigma: float = 2.0,
        **kwargs: Any
    ) -> None:
        """Plot trajectories with variance.

        Parameters
        ----------
        time : np.ndarray
            Time associated with the trajectory/trajectories.
        trajs : np.ndarray
            Trajectories to be plotted. Can be a single trajectory or list of trajectories.
        var : np.ndarray
            Variance along the trajectory/trajectories.
        sigma : np.ndarray, default=2
            How many standard deviations to visualize. By default, 2 standard deviations are shown.
        kwargs : Any
            Arguments to be passed into the plot function. Examples:
              - color, label, linestyle ...
            See the matplot lib documentation for more details
            https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html.
        """
        deviation = np.sqrt(var) * sigma
        plt.plot(time, trajs, **kwargs)
        if "color" in kwargs:
            plt.fill_between(
                time,
                trajs - deviation,
                trajs + deviation,
                alpha=0.1,
                color=kwargs["color"],
            )
        else:
            plt.fill_between(time, trajs - deviation, trajs + deviation, alpha=0.1)

    def plot_as_image(self, fig: matplotlib.figure.Figure) -> np.ndarray:
        """Generate image from plot figure.

        Parameters
        ----------
        fig : Figure
            Figure of interest.

        Returns
        -------
        np.ndarray
            Image as an array.
        """
        fig.canvas.draw()
        X_image = np.array(fig.canvas.renderer.buffer_rgba())
        return X_image.transpose(2, 0, 1)[0:3, :, :]

    def show(self) -> None:
        """Convenience method to show the plot."""
        plt.show()

    def viz_phase_xy(self, f: Callable, xs: np.ndarray, ys: np.ndarray) -> None:
        """Visualize the phase diagram in the xy plane.

        Parameters
        ----------
        f : Callable
            A callable function which takes in a numpy array of shape (B, 2).
            Where input[:, 0] is the x location and input[: 1] is the y location.
        xs : np.ndarray, shape=(X,)
            The x locations to plot the gradient.
        ys : np.ndarray, shape=(Y,)
            The y locations to plot the gradient.
        """
        # assumes f defines dynamics for a two dimentional state space

        # for s in states.
        XS, YS = np.meshgrid(xs, ys)
        grid_shape = XS.shape
        vector_input = np.hstack((XS.reshape(-1, 1), YS.reshape(-1, 1)))
        x_dot = f(vector_input)

        u = x_dot[:, 0].reshape(*grid_shape)
        v = x_dot[:, 1].reshape(*grid_shape)

        plt.quiver(XS, YS, u, v, np.arctan2(v, u))
        plt.xlabel("x")
        plt.ylabel("y")


PlotHandler = _PlotHandler()
