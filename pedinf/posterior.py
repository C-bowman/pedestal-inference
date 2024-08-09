from numpy import diff, isfinite, ndarray
from dataclasses import dataclass
from pedinf.diagnostics import SpectrometerModel
from inference.likelihoods import LogisticLikelihood
from inference.priors import BasePrior


class FlatPrior(BasePrior):
    def __call__(self, theta: ndarray) -> float:
        return 0.0

    def gradient(self, theta: ndarray) -> float:
        return 0.0

    def cost(self, theta: ndarray) -> float:
        return 0.0

    def cost_gradient(self, theta: ndarray) -> float:
        return 0.0


@dataclass
class SpectrumData:
    """
    Class to package spectral channel measurement data.
    """

    spectra: ndarray
    errors: ndarray
    spatial_channels: ndarray
    time_index: int
    shot: int

    def __post_init__(self):
        # check the data have the correct dimensions / shapes
        assert self.spectra.ndim == 2
        assert self.errors.ndim == 2
        assert self.spatial_channels.ndim == 1
        assert self.spectra.shape == self.errors.shape

        # check for any data which are NaN or inf
        assert self.spectra.shape[0] == self.spatial_channels.shape[0]
        bad_data = ~isfinite(self.spectra) | ~isfinite(self.errors)
        if bad_data.any():
            self.spectra[bad_data] = 0.0
            self.errors[bad_data] = 1e50

        # check validity of data values
        assert (self.errors > 0).all()
        assert (diff(self.spatial_channels) > 0).all()

        self.good_data = ~bad_data
        self.n_spectra = self.spectra.shape[1]


class ThomsonProfilePosterior:
    """
    Class for evaluating the posterior log-probability of the electron temperature and
    density profile parameters.

    :param spectrometer_model: \
        An instance of the ``SpectrometerModel`` class, which server as a forward-model
        for the polychromator measurements.

    :param spectrum_data: \
        An instance of the ``SpectrumData`` class, which packages the polychromator
        measurements, uncertainties and timing data.

    :param likelihood: \
        Either the ``GaussianLikelihood`` or ``LogisticLikelihood`` class from the
        ``inference.likelihoods`` module of the ``inference-tools`` package.

    :param prior: \
        An instance of a prior class from the ``inference.priors`` module, or a custom
        prior which inherits from the ``BasePrior`` base-class.
    """

    def __init__(
        self,
        spectrometer_model: SpectrometerModel,
        spectrum_data: SpectrumData,
        likelihood=LogisticLikelihood,
        prior: BasePrior = None,
    ):
        self.spectrum = spectrometer_model
        self.data = spectrum_data

        # build the likelihood function for spectral measurements
        self.likelihood = likelihood(
            y_data=self.data.spectra.flatten(),
            sigma=self.data.errors.flatten(),
            forward_model=self.spectrum.predictions,
            forward_model_jacobian=self.spectrum.predictions_jacobian,
        )

        self.prior = FlatPrior() if prior is None else prior

    def __call__(self, theta: ndarray) -> float:
        return self.likelihood(theta) + self.prior(theta)

    def gradient(self, theta: ndarray) -> ndarray:
        return self.likelihood.gradient(theta) + self.prior.gradient(theta)

    def cost(self, theta: ndarray) -> float:
        return -self.likelihood(theta) - self.prior(theta)

    def cost_gradient(self, theta: ndarray) -> ndarray:
        return -self.likelihood.gradient(theta) - self.prior.gradient(theta)
