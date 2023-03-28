from numpy import linspace, empty_like, zeros, allclose
from numpy.random import default_rng
from pedinf.models import ProfileModel, mtanh, lpm
import pytest


test_models = [lpm, mtanh]
parameter_test_ranges = {
    "pedestal_location": (1.3, 1.45),
    "pedestal_height": (50., 300.),
    "pedestal_width": (0.005, 0.02),
    "pedestal_top_gradient": (-100., 1000.),
    "background_level": (0.5, 15.),
    "logistic_shape_parameter": (0.1, 3),
}

rng = default_rng(123)


def test_param_ranges():
    for model in test_models:
        assert all(p in parameter_test_ranges for p in model.parameters)


def build_test_data(model: ProfileModel, n_points=10):
    R = linspace(1.2, 1.5, 256)
    theta = zeros([n_points, model.n_parameters])
    for i, p in enumerate(model.parameters):
        theta[:, i] = rng.uniform(*parameter_test_ranges[p], size=n_points)
    return R, theta


@pytest.mark.parametrize("model", test_models)
def test_jacobian(model: ProfileModel):
    R, theta = build_test_data(model)
    delta = 1e-6

    for t in theta:
        prediction_1 = model.prediction(R, t)
        jacobian_1 = model.jacobian(R, t)
        prediction_2, jacobian_2 = model.prediction_and_jacobian(R, t)

        assert allclose(prediction_1, prediction_2)
        assert allclose(jacobian_1, jacobian_2)

        fd_jac = empty_like(jacobian_1)
        for i in range(jacobian_1.shape[1]):
            t1, t2 = t.copy(), t.copy()
            dt = t[i] * delta
            t1[i] -= dt
            t2[i] += dt
            df = model.prediction(R, t2) - model.prediction(R, t1)
            fd_jac[:, i] = 0.5 * df / dt

        error = abs(jacobian_1 - fd_jac) / abs(fd_jac).max()
        assert error.max() < 1e-6


@pytest.mark.parametrize("model", test_models)
def test_gradient(model: ProfileModel):
    R, theta = build_test_data(model)
    dR = 1e-5
    for t in theta:
        grad = model.gradient(R, t)
        fd = (model.prediction(R + dR, t) - model.prediction(R - dR, t)) * (0.5 / dR)
        error = abs(grad - fd) / abs(fd).max()
        assert error.max() < 1e-4
