from numpy import array
from numpy.random import normal
from scipy.optimize import fmin_l_bfgs_b, differential_evolution
from itertools import product
from pedinf.model import PedestalModel

from inference.likelihoods import LogisticLikelihood
from inference.priors import UniformPrior, GaussianPrior, ExponentialPrior, JointPrior
from inference.posterior import Posterior
from inference.mcmc import EnsembleSampler


def edge_profile_sample(radius, y_data, y_err, n_samples=10000, walkers=500, plot_diagnostics=False):
    posterior = PedestalPosterior(
        x=radius,
        y=y_data,
        y_err=y_err
    )
    theta_mode = posterior.locate_mode()

    # protection against the mode being zero
    for i in range(theta_mode.size):
        if theta_mode[i] == 0.:
            l, u = posterior.bounds[i]
            theta_mode[i] = 0.05 * (u - l)

    # setup ensemble sampling
    starts = [theta_mode * (1 + 0.02 * normal(size=theta_mode.size)) for _ in range(walkers)]
    chain = EnsembleSampler(
        posterior=posterior.posterior,
        starting_positions=starts
    )
    # run the sampler
    d, r = divmod(n_samples, walkers)
    iterations = d if r == 0 else d + 1
    chain.advance(iterations=iterations + 50)
    sample = chain.get_sample(burn=walkers * 50)[:n_samples, :]

    if plot_diagnostics:
        chain.plot_diagnostics()
        chain.trace_plot()

    return sample


class PedestalPosterior(object):
    def __init__(self, x, y, y_err, likelihood=LogisticLikelihood):
        self.x = x
        self.y = y
        self.sigma = y_err
        self.model = PedestalModel(R=x)

        self.bounds = [
            (self.x.min(), self.x.max()),
            (0., self.y.max()*1.5),
            (self.x.ptp()*1e-3, self.x.ptp()),
            (0., 2.),
            (0., 0.05),
            (-5, 1.)
        ]

        self.likelihood = likelihood(
            y_data=self.y,
            sigma=self.sigma,
            forward_model=self.model.prediction,
            forward_model_jacobian=self.model.jacobian
        )

        self.prior = JointPrior(
            components=[
                UniformPrior(
                    lower=[b[0] for b in self.bounds[:4]],
                    upper=[b[1] for b in self.bounds[:4]],
                    variable_indices=[0, 1, 2, 3]
                ),
                ExponentialPrior(beta=5e-3, variable_indices=[4]),
                GaussianPrior(mean=0., sigma=0.5, variable_indices=[5])
            ],
            n_variables=6
        )
        self.posterior = Posterior(likelihood=self.likelihood, prior=self.prior)

    def initial_guesses(self):
        dx, dy = self.x.ptp(), self.y.ptp()
        guesses = product(
            dx * array([0.3, 0.5, 0.7]) + self.x.min(),
            self.y.max() * array([0.3, 0.5, 0.7]),
            dx * array([0.07, 0.15, 0.3]),
            [0.2],
            [0.01],
            [-0.35, 0.01, 0.35]
        )
        return [array(g) for g in guesses]

    def locate_mode(self):
        guesses = sorted(self.initial_guesses(), key=self.likelihood)
        bfgs_mode, fmin, D = fmin_l_bfgs_b(
            func=self.posterior.cost,
            fprime=self.posterior.cost_gradient,
            x0=guesses[-1],
            bounds=self.bounds
        )

        de_result = differential_evolution(
            func=self.posterior.cost,
            bounds=self.bounds
        )

        return bfgs_mode if fmin < de_result.fun else de_result.x