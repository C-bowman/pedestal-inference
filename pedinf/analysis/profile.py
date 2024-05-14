from matplotlib import pyplot as plt
from numpy import ndarray, array

from pedinf.analysis.utils import vectorised_hdi
from pedinf.models import ProfileModel


class PlasmaProfile:
    """
    Class for representing the estimate of a plasma profile derived from
    samples of possible profiles produced by uncertainty quantification.

    :param axis: \
        The axis values at which the profile samples were evaluated as
        a 1D ``numpy.ndarray``.

    :param profile_samples: \
        The profile samples as a 2D ``numpy.ndarray`` of shape ``(axis.size, n_samples)``.

    :param axis_label: \
        A description of the axis on which the profiles have been evaluated,
        e.g. 'major radius'.

    :param profile_label: \
        A description of the quantity represented by the profile,
        e.g. 'electron temperature'.

    :param axis_units: \
        The units of the axis values, e.g. 'm' for metres.

    :param profile_units: \
        The units of the profile values, e.g. 'eV' for electron temperature.
    """
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
        self.axis_label = "profile axis" if axis_label is None else axis_label
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
        assert self.profile_samples.shape[1] > 3

        self.hdi_65 = vectorised_hdi(self.profile_samples.T, frac=0.65)
        self.hdi_95 = vectorised_hdi(self.profile_samples.T, frac=0.95)
        self.mean = self.profile_samples.mean(axis=1)

    @classmethod
    def from_parameters(
        cls,
        axis: ndarray,
        parameter_samples: ndarray,
        model: ProfileModel,
        gradient: bool = False,
        axis_label: str = None,
        profile_label: str = None,
        axis_units: str = None,
        profile_units: str = None,
    ):
        """
        Class for representing the estimate of a plasma profile derived from
        samples of possible profiles produced by uncertainty quantification.

        :param axis: \
            The axis values at which the profile samples are to be evaluated as
            a 1D ``numpy.ndarray``.

        :param parameter_samples: \
            The model parameter samples as a 2D ``numpy.ndarray`` of shape
            ``(model.n_parameters, n_samples)``.

        :param model: \
            An instance of a profile model class from the ``pedinf.models`` module.

        :param gradient: \
            If given ``True`` the profile samples will be the gradient of the model
            prediction instead of the prediction itself. Default is ``False``.

        :param axis_label: \
            A description of the axis on which the profiles have been evaluated,
            e.g. 'major radius'.

        :param profile_label: \
            A description of the quantity represented by the profile,
            e.g. 'electron temperature'.

        :param axis_units: \
            The units of the axis values, e.g. 'm' for metres.

        :param profile_units: \
            The units of the profile values, e.g. 'eV' for electron temperature.
        """

        assert parameter_samples.ndim == 2
        assert parameter_samples.shape[0] == model.n_parameters
        # make a copy of the original model to avoid altering its internal state
        model = model.from_configuration(model.get_model_configuration())
        model.update_radius(axis)
        if gradient:
            profile_samples = array(
                [model.forward_gradient(s) for s in parameter_samples.T]
            ).T
        else:
            profile_samples = array(
                [model.forward_prediction(s) for s in parameter_samples.T]
            ).T
        assert profile_samples.shape == (axis.size, parameter_samples.shape[1])
        return cls(
            axis=axis,
            profile_samples=profile_samples,
            axis_label=axis_label,
            profile_label=profile_label,
            axis_units=axis_units,
            profile_units=profile_units
        )

    def plot(self, axis=None, color: str = None):
        """
        Plot the profile.

        :param axis: \
            A `matplotlib` axis object on which the profile will be plotted.

        :param color: \
            A valid ``matplotlib`` color string which will be used to plot the profile.
        """
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
            label="95% HDI",
        )
        ax.fill_between(
            self.axis,
            self.hdi_65[:, 0],
            self.hdi_65[:, 1],
            color=col,
            alpha=0.25,
            label="65% HDI",
        )
        ax.plot(self.axis, self.mean, color=col, ls="dashed", lw=2, label="mean")
        ax.set_ylabel(self._ylabel)
        ax.set_xlabel(self._xlabel)
        ax.grid()
        ax.legend()

        if axis is None:
            plt.show()

    def __str__(self):
        return f"""\n
        \r[ PlasmaProfile object ]
        \r>> profile label: {self.profile_label}
        \r>> profile units: {self.profile_units if self.profile_units != "" else "none"}
        \r>>    axis label: {self.axis_label}
        \r>>    axis units: {self.axis_units if self.axis_units != "" else "none"}
        \r>>    axis range: {self.axis.min()} -> {self.axis.max()}
        """
