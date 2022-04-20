from numpy import linspace, array, empty_like, zeros
from numpy.random import default_rng
from pedinf.model import PedestalModel, mtanh, mtanh_gradient
from pedinf import PedestalPosterior
import pytest


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


@pytest.fixture()
def data():
    R = linspace(0, 1, 200)
    model = PedestalModel(R=R)
    theta = array([0.5, 2., 0.2, 0.2, 0.01, -0.5])
    return R, model, theta


def test_jacobian(data):
    _, model, theta = data
    jacobian = model.jacobian(theta)
    fd_jac = empty_like(jacobian)

    delta = 1e-6
    for i in range(jacobian.shape[1]):
        t1, t2 = theta.copy(), theta.copy()
        dt = theta[i] * delta
        t1[i] -= dt
        t2[i] += dt
        f1 = model.prediction(t1)
        f2 = model.prediction(t2)

        df = 0.5 * (f2 - f1) / dt
        fd_jac[:, i] = df

    error = abs_frac_error(fd_jac, jacobian)
    assert error.max() < 1e-5


def test_gradient(data):
    R, _, theta = data
    dR = 1e-4
    grad = mtanh_gradient(R, theta)
    df = mtanh(R + dR, theta) - mtanh(R - dR, theta)

    error = abs_frac_error(0.5 * df / dR, grad)
    assert error.max() < 1e-5


def test_likelihood(data):
    rng = default_rng(256)
    theta = [1.45, 100., 0.03, 0.2, 0.01, 0.75]

    # create synthetic testing data
    R_data = linspace(1.35, 1.53, 19)
    Te_data = mtanh(R_data, theta)
    Te_err = 2 + Te_data * 0.05
    Te_data = abs(Te_data + Te_err * rng.normal(size=Te_data.size))
    posterior = PedestalPosterior(x=R_data, y=Te_data, y_err=Te_err)

    guesses = posterior.initial_guesses()
    for g in guesses:
        grad = posterior.likelihood.gradient(g)
        fd_grad = finite_difference(posterior.likelihood, g, vectorised_arguments=True)
        error = abs_frac_error(grad, fd_grad)
        assert (error < 1e-3).all()
