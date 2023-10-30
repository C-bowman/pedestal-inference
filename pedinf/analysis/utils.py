from numpy import isfinite, sort, expand_dims, take_along_axis
from numpy import atleast_1d, linspace, zeros, ndarray
import matplotlib.pyplot as plt
from warnings import warn
from pedinf.models import ProfileModel


def locate_radius(
    profile_values: ndarray,
    parameters: ndarray,
    model: ProfileModel,
    search_limits=(1.2, 1.5),
    tolerance=1e-4,
    search_points: int = 25,
    max_newton_updates: int = 8,
    show_warnings: bool = True,
):
    """
    For a given edge profile, find the radius values at which the profile
    is equal to a given set of target values.

    :param profile_values: \
        The target profile values for which the radius will be calculated.

    :param parameters: \
        The parameters of the profile model.

    :param model: \
        An instance of one of the model classes from ``pedinf.models``.

    :param search_limits: \
        A tuple of two floats specifying the lower and upper limits
        of radius values when searching for the target profile values.

    :param tolerance: \
        The convergence threshold for the absolute fractional error in
        the target profile value.

    :param search_points: \
        The number of points used in the initial grid-search which
        generates initial guesses for Newton iteration.

    :param search_points: \
        The number of points used in the initial grid-search which
        generates initial guesses for Newton iteration.

    :param max_newton_updates: \
        The maximum allowed number of Newton iterations used when
        finding the radii of the target profile values.

    :param show_warnings: \
        Whether to display warnings regarding failure of Newton update
        convergence.
    """
    targets = atleast_1d(profile_values)
    if (targets <= 0.0).any() or not isfinite(targets).all():
        raise ValueError(
            f"""\n
            [ locate_radius error ]
            >> All values given in the 'profile_values' argument must be
            >> finite and greater than zero.
            """
        )
    # evaluate the profile on the search grid
    R_min, R_max = search_limits
    R_search = linspace(R_min, R_max, search_points)
    profile = model.prediction(R_search, parameters)
    cut = profile.argmax()
    R_search = R_search[cut:]
    profile = profile[cut:]

    # find the points closest to each target profile value
    indices = abs(targets[:, None] - profile[None, :]).argmin(axis=1)
    R = R_search[indices]
    initial_R = R.copy()
    initial_val = profile[indices]
    initial_err = abs((targets - initial_val) / targets)

    if not isfinite(initial_val).all():
        raise ValueError(
            f"""\n
            [ locate_radius error ]
            >> The initial estimate of the radius values produced by the grid
            >> search contains non-finite profile values.
            """
        )

    for i in range(max_newton_updates):
        dy = targets - model.prediction(R, parameters)
        R += dy / model.gradient(R, parameters)
        R.clip(min=R_min, max=R_max, out=R)
        error = abs(dy / targets)
        if (error < tolerance).all():
            break
    else:
        # if any points didn't meet the tolerance, check if their error diverged
        diverged = error > initial_err
        if diverged.any():  # replace any which diverged with the grid-search result
            R[diverged] = initial_R[diverged]
        if show_warnings:
            warn(
                f"""\n
                [ locate_radius warning ]
                >> Newton iteration failed to converge for {(error > tolerance).sum()} out
                >> of {targets.size} of the target profile values.
                """
            )
    return R


def vectorised_hdi(samples: ndarray, frac: float) -> ndarray:
    s = sort(samples, axis=0)
    n, m = samples.shape
    L = int(frac * n)

    # check that we have enough samples to estimate the HDI for the chosen fraction
    hdi = zeros([m, 2])
    if n > L:
        # find the optimal single HDI
        widths = s[L:, :] - s[: n - L, :]
        i = expand_dims(widths.argmin(axis=0), axis=0)
        hdi[:, 0] = take_along_axis(s, i, 0).squeeze()
        hdi[:, 1] = take_along_axis(s, i + L, 0).squeeze()
    else:
        hdi[:, 0] = s[0, :]
        hdi[:, 1] = s[-1, :]
    return hdi


class PlasmaProfile:
    def __init__(
        self,
        axis: ndarray,
        profile_samples: ndarray,
        axis_label: str = None,
        profile_label: str = None,
        axis_units: str = None,
        profile_units: str = None,
    ):
        self.axis = axis
        self.profile_samples = profile_samples
        self.axis_label = "x-axis" if axis_label is None else axis_label
        self.profile_label = "profile value" if profile_label is None else profile_label
        self.axis_units = "" if axis_units is None else axis_units
        self.profile_units = "" if profile_units is None else profile_units

        x_unit = "" if axis_units is None else f" ({axis_units})"
        y_unit = "" if profile_units is None else f" ({profile_units})"
        self._xlabel = self.axis_label + x_unit
        self._ylabel = self.profile_label + y_unit

        assert self.axis.ndim == 1
        assert self.profile_samples.ndim == 2
        assert self.profile_samples.shape[0] == self.axis.size

        self.hdi_65 = vectorised_hdi(self.profile_samples.T, frac=0.65)
        self.hdi_95 = vectorised_hdi(self.profile_samples.T, frac=0.95)
        self.mean = self.profile_samples.mean(axis=1)

    def plot(self, axis=None, color=None):
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = axis

        col = "blue" if color is None else color

        ax.fill_between(
            self.axis, self.hdi_95[:, 0], self.hdi_65[:, 0], color=col, alpha=0.1
        )
        ax.fill_between(
            self.axis,
            self.hdi_65[:, 1],
            self.hdi_95[:, 1],
            color=col,
            alpha=0.1,
            label="65% HDI",
        )
        ax.fill_between(
            self.axis,
            self.hdi_65[:, 0],
            self.hdi_65[:, 1],
            color=col,
            alpha=0.25,
            label="95% HDI",
        )
        ax.plot(self.axis, self.mean, color=col, ls="dashed", lw=2, label="mean")
        ax.set_ylabel(self._ylabel)
        ax.set_xlabel(self._xlabel)
        ax.grid()
        ax.legend()

        if axis is None:
            plt.show()
