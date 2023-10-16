from numpy import array, diff, isfinite, ndarray
from numpy.random import normal
from scipy.optimize import fmin_l_bfgs_b, differential_evolution
from dataclasses import dataclass
from itertools import product
from pedinf.models import ProfileModel, lpm
from pedinf.diagnostics import SpectrometerModel

from inference.likelihoods import LogisticLikelihood
from inference.priors import UniformPrior, GaussianPrior, ExponentialPrior, JointPrior
from inference.posterior import Posterior
from inference.mcmc import EnsembleSampler


def edge_profile_sample(
    radius: ndarray,
    y_data: ndarray,
    y_err: ndarray,
    n_samples=10000,
    n_walkers=500,
    plot_diagnostics=False,
):
    """
    Generates a sample of possible edge profiles given Thomson-scattering
    measurements of the plasma edge.

    :param radius: \
        The major radius values of the Thomson-scattering measurements
        as a 1D ``numpy.ndarray``.

    :param y_data: \
        The measured profile values as 1D ``numpy.ndarray``.

    :param y_err: \
        The uncertainty of the measured profile values as 1D ``numpy.ndarray``.

    :param n_samples: \
        The total number of samples which will be generated.

    :param n_walkers: \
        The number of 'walkers' used in the sampling process. See the
        documentation for the ``EnsembleSampler`` class in the
        ``inference-tools`` package for further details.

    :param bool plot_diagnostics: \
        Selects whether diagnostic plots for the sampling a displayed.

    :return: \
        The samples as a 2D ``numpy.ndarray`` array of shape ``(n_samples, n_parameters)``.
    """
    # filter out any points which aren't finite
    finite = isfinite(y_data) & isfinite(y_err)
    if finite.sum() < 3:
        raise ValueError(
            """
            [ edge_profile_sample error ]
            >> The 'y_data' and 'y_err' arrays do not contain enough
            >> finite values (< 3) to proceed with the analysis.
            """
        )

    posterior = PedestalPosterior(
        x=radius[finite], y=y_data[finite], y_err=y_err[finite]
    )
    theta_mode = posterior.locate_mode()

    # setup ensemble sampling
    starts = [
        theta_mode * normal(size=theta_mode.size, loc=1, scale=0.02)
        for _ in range(n_walkers)
    ]
    chain = EnsembleSampler(
        posterior=posterior.posterior, starting_positions=starts, display_progress=False
    )
    # run the sampler
    d, r = divmod(n_samples, n_walkers)
    iterations = d if r == 0 else d + 1
    burn_itr = 70
    chain.advance(iterations=iterations + burn_itr)
    sample = chain.get_sample(burn=n_walkers * burn_itr)[:n_samples, :]

    if plot_diagnostics:
        chain.plot_diagnostics()
        chain.trace_plot()

    return sample


class PedestalPosterior:
    def __init__(self, x, y, y_err, likelihood=LogisticLikelihood):
        self.x = x
        self.y = y
        self.sigma = y_err
        self.model = lpm

        ymax = self.y.max()
        self.bounds = [
            (self.x.min(), self.x.max()),
            (ymax * 0.05, ymax * 1.5),
            (self.x.ptp() * 1e-2, self.x.ptp()),
            (-ymax * 5, ymax * 20),
            (ymax * 1e-3, ymax * 0.05),
            (-2, 1.0),
        ]

        self.likelihood = likelihood(
            y_data=self.y,
            sigma=self.sigma,
            forward_model=self.model.prediction,
            forward_model_jacobian=self.model.jacobian,
        )

        self.prior = JointPrior(
            components=[
                UniformPrior(
                    lower=[b[0] for b in self.bounds[:4]],
                    upper=[b[1] for b in self.bounds[:4]],
                    variable_indices=[0, 1, 2, 3],
                ),
                ExponentialPrior(beta=5e-3 * ymax, variable_indices=[4]),
                GaussianPrior(mean=0.0, sigma=0.5, variable_indices=[5]),
            ],
            n_variables=6,
        )
        self.posterior = Posterior(likelihood=self.likelihood, prior=self.prior)

    def initial_guesses(self):
        dx, dy = self.x.ptp(), self.y.ptp()
        guesses = product(
            dx * array([0.3, 0.5, 0.7]) + self.x.min(),
            self.y.max() * array([0.3, 0.5, 0.7]),
            dx * array([0.07, 0.15, 0.3]),
            [dy / dx],
            [0.01 * self.y.max()],
            [-0.35, 0.01, 0.35],
        )
        return [array(g) for g in guesses]

    def locate_mode(self):
        guesses = sorted(self.initial_guesses(), key=self.likelihood)
        bfgs_mode, fmin, D = fmin_l_bfgs_b(
            func=self.posterior.cost,
            fprime=self.posterior.cost_gradient,
            x0=guesses[-1],
            bounds=self.bounds,
        )

        de_result = differential_evolution(func=self.posterior.cost, bounds=self.bounds)

        return bfgs_mode if fmin < de_result.fun else de_result.x


class FlatPrior:
    def __call__(self, theta):
        return 0.

    def gradient(self, theta):
        return 0.


@dataclass
class SpectrumData:
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


class SpectralPedestalPosterior:
    def __init__(
        self,
        spectrometer_model: SpectrometerModel,
        spectrum_data: SpectrumData,
        likelihood=LogisticLikelihood,
        prior=None,
    ):
        self.spectrum = spectrometer_model
        self.data = spectrum_data

        self.likelihood = likelihood(
            y_data=self.data.spectra.flatten(),
            sigma=self.data.errors.flatten(),
            forward_model=self.spectrum.predictions,
            forward_model_jacobian=self.spectrum.predictions_jacobian
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