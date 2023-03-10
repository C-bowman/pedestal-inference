from numpy import linspace, array, empty_like, zeros
from numpy.random import default_rng
from pedinf.models import ProfileModel, mtanh, lpm
import pytest


test_models = [lpm]
parameter_test_ranges = {
    "R0 (pedestal location)": (1.3, 1.45),
    "h (pedestal height)": (50., 300.),
    "w (pedestal width)": (0.005, 0.02),
    "a (pedestal top gradient)": (-100., 1000.),
    "b (background level)": (0.5, 15.),
    "ln_k (logistic shape parameter)": (-2, 1),
}

rng = default_rng(123)


def test_param_ranges():
    for model in test_models:
        assert all(p in parameter_test_ranges for p in model.parameters)


def finite_difference(func=None, x0=None, delta=1e-5, vectorised_arguments=False):
    grad = zeros(x0.size)
    for i in range(x0.size):
        x1 = x0.copy()
        x2 = x0.copy()
        dx = x0[i] * delta

        x1[i] -= dx
        x2[i] += dx

        if vectorised_arguments:
            f1 = func(x1)
            f2 = func(x2)
        else:
            f1 = func(*x1)
            f2 = func(*x2)

        grad[i] = 0.5 * (f2 - f1) / dx
    return grad


def abs_frac_error(a, b):
    zero_inds = (a == 0.0) & (b == 0.0)
    af_err = abs(a / b - 1.0)
    af_err[zero_inds] = 0.0
    return af_err


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
        jacobian = model.jacobian(R, t)
        fd_jac = empty_like(jacobian)
        for i in range(jacobian.shape[1]):
            t1, t2 = t.copy(), t.copy()
            dt = t[i] * delta
            t1[i] -= dt
            t2[i] += dt
            f1 = model.prediction(R, t1)
            f2 = model.prediction(R, t2)

            df = 0.5 * (f2 - f1) / dt
            fd_jac[:, i] = df

        error = abs(jacobian - fd_jac) / abs(fd_jac).max()
        assert error.max() < 1e-4


@pytest.mark.parametrize("model", test_models)
def test_gradient(model: ProfileModel):
    R, theta = build_test_data(model)
    dR = 1e-5
    for t in theta:
        grad = model.gradient(R, t)
        fd = (model.prediction(R + dR, t) - model.prediction(R - dR, t)) * (0.5 / dR)
        error = abs(grad - fd) / abs(fd).max()
        assert error.max() < 1e-4
