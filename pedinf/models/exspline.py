from typing import Tuple
from functools import partial
from numpy import exp, ndarray, zeros
from pedinf.models import ProfileModel
from pedinf.models.utils import b_spline_basis


class exspline(ProfileModel):
    name = "exspline"

    def __init__(self, knots: ndarray, radius=None, low_field_side=True):
        self.knots = knots
        self.drn = -1 if low_field_side else 1

        self.n_parameters = 3 + self.knots.size
        self.parameters = {
            "logistic_location": 0,
            "logistic_floor": 1,
            "logistic_width": 2,
            "basis_weights": slice(3, self.n_parameters)
        }

        if radius is not None:
            self.update_radius(self, radius)

    def update_radius(self, radius: ndarray):
        self.radius = radius
        self.basis, self.derivs = b_spline_basis(self.radius, self.knots, derivatives=True)
        self.forward_prediction = partial(self._prediction, self.radius, self.basis)
        self.forward_gradient = partial(self._gradient, self.radius, self.derivs)
        self.forward_jacobian = partial(self._jacobian, self.radius, self.basis)
        self.forward_prediction_and_jacobian = partial(self._prediction_and_jacobian, self.radius, self.basis)

    def prediction(self, radius: ndarray, theta: ndarray) -> ndarray:
        basis = b_spline_basis(radius, self.knots)
        return self._prediction(radius, basis, theta)

    def gradient(self, radius: ndarray, theta: ndarray) -> ndarray:
        basis, derivs = b_spline_basis(radius, self.knots, derivatives=True)
        return self._gradient(radius, basis, derivs, theta)

    def jacobian(self, radius: ndarray, theta: ndarray) -> ndarray:
        basis = b_spline_basis(radius, self.knots)
        return self._jacobian(radius, basis, theta)

    def prediction_and_jacobian(self, radius: ndarray, theta: ndarray) -> Tuple[ndarray, ndarray]:
        basis = b_spline_basis(radius, self.knots)
        return self._prediction_and_jacobian(radius, basis, theta)

    def _prediction(self, radius: ndarray, basis: ndarray, theta: ndarray) -> ndarray:
        z = (4 * self.drn) * (radius - theta[0]) / theta[2]
        logistic = (1 - theta[1]) / (1 + exp(-z)) + theta[1]
        exp_spline = exp(basis @ theta[3:])
        return logistic * exp_spline

    def _jacobian(self, radius: ndarray, basis: ndarray, theta: ndarray):
        exp_spline = exp(basis @ theta[3:])
        z = (4 * self.drn) * (radius - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - theta[1])
        dy_df = (1 - L) * exp_spline
        dy_dz = q * dy_df
        y = (q + theta[1]) * exp_spline

        jac = zeros([radius.size, self.n_parameters])
        jac[:, 0] = -dy_dz * (4 * self.drn / theta[2])
        jac[:, 1] = dy_df
        jac[:, 2] = -dy_dz * z / theta[2]
        jac[:, self.parameters["basis_weights"]] = basis * y[:, None]
        return jac

    def _prediction_and_jacobian(self, radius: ndarray, basis: ndarray, theta: ndarray):
        exp_spline = exp(basis @ theta[3:])
        z = (4 * self.drn) * (radius - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - theta[1])
        dy_df = (1 - L) * exp_spline
        dy_dz = q * dy_df
        y = (q + theta[1]) * exp_spline

        jac = zeros([radius.size, self.n_parameters])
        jac[:, 0] = -dy_dz * (4 * self.drn / theta[2])
        jac[:, 1] = dy_df
        jac[:, 2] = -dy_dz * z / theta[2]
        jac[:, self.parameters["basis_weights"]] = basis * y[:, None]
        return y, jac

    def _gradient(self, radius: ndarray, basis: ndarray, basis_derivs: ndarray, theta: ndarray):
        s = exp(basis @ theta[3:])
        ds_dr = basis_derivs @ theta[3:]
        z = (4 * self.drn) * (radius - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - theta[1])
        g = q + theta[1]
        y = g * s
        return y * (basis_derivs @ theta[3:]) + s * q * (1 - L) * (4 * self.drn / theta[2])
