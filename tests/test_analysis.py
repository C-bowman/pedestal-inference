from numpy import array
from numpy.random import default_rng
from pedinf.models import lpm, mtanh, ProfileModel
from typing import Type
from pedinf.analysis import locate_radius, pressure_parameters
import pytest


def test_locate_radius():
    theta = array([1.4, 150, 0.02, 600, 5, 0.75])
    for value in [30., 40., 50., 60.]:
        R = locate_radius(profile_value=value, theta=theta, model=lpm)
        assert abs(lpm.prediction(R, theta) / value - 1.).max() < 1e-8


@pytest.mark.parametrize("model", [mtanh, lpm])
def test_pressure_parameters(model: Type[ProfileModel]):
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