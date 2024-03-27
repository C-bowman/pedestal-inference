from numpy import array, linspace
from numpy.random import default_rng
from pedinf.models import lpm, mtanh, ProfileModel
from pedinf.analysis.utils import locate_radius
from pedinf.analysis import pressure_parameters, pressure_profile_and_gradient
from pedinf.analysis import PlasmaProfile
import pytest


def test_locate_radius():
    theta = array([1.4, 150, 0.02, 600, 5, 0.5])
    temperatures = array([10., 15., 20., 30., 40., 50., 60.])
    model = lpm()
    R = locate_radius(profile_values=temperatures, parameters=theta, model=model)
    assert abs(model.prediction(R, theta) / temperatures - 1.).max() < 1e-8


@pytest.mark.parametrize("model", [mtanh(), lpm()])
def test_pressure_parameters(model: ProfileModel):
    # set bounds to randomly sample te / ne profile parameters
    rng = default_rng(1)
    ne_lwr = [1.35, 2e19, 0.004, -1e19, 1e17, 0.6]
    ne_upr = [1.4, 8e19, 0.02, 1e19, 1e18, 1.1]
    te_lwr = [1.35, 100, 0.006, 0., 1, 0.75]
    te_upr = [1.4, 250, 0.025, 3000, 10, 1.1]

    for i in range(50):
        ne = rng.uniform(low=ne_lwr, high=ne_upr)
        te = rng.uniform(low=te_lwr, high=te_upr)
        te[0] = ne[0] + rng.normal(scale=0.01)
        theta, info = pressure_parameters(
            ne_parameters=ne[:model.n_parameters],
            te_parameters=te[:model.n_parameters],
            model=model,
            return_diagnostics=True
        )
        assert info["max_abs_err"] / info["target"].mean() < 0.2


def test_pressure_profile_and_gradient():
    n_samples = 250
    rng = default_rng(2)
    te_means = array([1.38, 150., 0.03, 400., 10.])
    te_sigma = array([0.005, 10., 0.005, 50., 3.])
    ne_means = array([1.375, 3e19, 0.02, 1e19, 2e18])
    ne_sigma = array([0.005, 3e18, 0.003, 1e19, 3.])

    te_samples = rng.normal(loc=te_means, scale=te_sigma, size=[n_samples, 5])
    ne_samples = rng.normal(loc=ne_means, scale=ne_sigma, size=[n_samples, 5])
    R = linspace(1.2, 1.5, 256)
    pe_profile, pe_grad_profile = pressure_profile_and_gradient(
        radius=R,
        te_profile_samples=te_samples,
        ne_profile_samples=ne_samples,
        model=mtanh()
    )

    assert isinstance(pe_profile, PlasmaProfile)
    assert isinstance(pe_grad_profile, PlasmaProfile)
