from typing import Tuple, Sequence
from types import ModuleType
from functools import partial
from numpy import ndarray
from pedinf.models import ProfileModel
from pedinf.models.utils import b_spline_basis


class exspline(ProfileModel):
    r"""
    'exspline' uses a combination of exponentiated b-splines and a logistic function to model
    the profile shape. The logistic function is combined with the exponentiated splines
    multiplicatively, and so models the pedestal as a localised fractional decrease in the
    background profile. The size of the fractional decrease is controlled by the parameter
    :math:`f \in [0, 1]`.

    .. math::

       \mathrm{exspline}(R, \, \underline{\theta}) = \exp{\left[\sum_{i=1}^{n} a_i \phi_i(R)\right]}
       \left((1 - f) L(z) + f\right)

    where

    .. math::

       z = -4 \frac{R - R_0}{w}, \quad \quad L(x) = \frac{1}{1 + e^{-x}}

    The model parameter vector :math:`\underline{\theta}` has the following order:

    .. math::

       \underline{\theta} = \left[ \,  R_0, \, f, \, w, \, a_1 \, a_2, \, \ldots, \, a_n \, \right],

    where

     - :math:`R_0` is the logistic function location.
     - :math:`f` is the logistic function 'floor'.
     - :math:`w` is the logistic function width.
     - :math:`a_i` is the weight for the :math:`i`'th b-spline basis function.
    """
    name = "exspline"

    def __init__(self, knots: ndarray, radius=None, low_field_side=True, jit_compile=True):
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
            self.update_radius(radius)

        if jit_compile:
            import jax.numpy as jnp
            from jax import jit
            methods = [jit(m) for m in build_methods(lib=jnp)]
        else:
            import numpy as np
            methods = build_methods(lib=np)

        (
            self._prediction,
            self._jacobian,
            self._prediction_and_jacobian,
            self._gradient
        ) = methods

    def update_radius(self, radius: ndarray):
        self.radius = radius
        self.basis, self.derivs = b_spline_basis(self.radius, self.knots, derivatives=True)
        self.forward_prediction = partial(self._prediction, self.radius, self.basis,  self.drn)
        self.forward_gradient = partial(self._gradient, self.radius, self.basis, self.derivs,  self.drn)
        self.forward_jacobian = partial(self._jacobian, self.radius, self.basis,  self.drn)
        self.forward_prediction_and_jacobian = partial(self._prediction_and_jacobian, self.radius, self.basis,  self.drn)

    def prediction(self, radius: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the prediction of the ``exspline`` model.
        See the documentation for ``exspline`` for details of the model itself.

        :param radius: \
            Radius values at which the prediction is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The predicted profile at the given radius values.
        """
        basis = b_spline_basis(radius, self.knots)
        return self._prediction(radius, basis, self.drn, theta)

    def gradient(self, radius: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the gradient (w.r.t. major radius) of the ``exspline`` model.
        See the documentation for ``exspline`` for details of the model itself.

        :param radius: \
            Radius values at which the gradient is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The predicted gradient profile at the given radius values.
        """
        basis, derivs = b_spline_basis(radius, self.knots, derivatives=True)
        return self._gradient(radius, basis, derivs, self.drn, theta)

    def jacobian(self, radius: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the jacobian of the ``exspline`` model. The jacobian is a matrix where
        element :math:`i, j` is the derivative of the model prediction at the
        :math:`i`'th radial position with respect to the :math:`j`'th model parameter.
        See the documentation for ``exspline`` for details of the model itself.

        :param radius: \
            Radius values at which the jacobian is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The jacobian matrix for the given radius values.
        """
        basis = b_spline_basis(radius, self.knots)
        return self._jacobian(radius, basis, self.drn, theta)

    def prediction_and_jacobian(self, radius: ndarray, theta: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Calculates the prediction and the jacobian of the ``exspline`` model. The jacobian
        is a matrix where element :math:`i, j` is the derivative of the model prediction
        at the :math:`i`'th radial position with respect to the :math:`j`'th model parameter.
        See the documentation for ``exspline`` for details of the model itself.

        :param radius: \
            Radius values at which the prediction and jacobian are evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The model prediction and the jacobian matrix for the given radius values.
        """
        basis = b_spline_basis(radius, self.knots)
        return self._prediction_and_jacobian(radius, basis, self.drn, theta)

    def get_model_configuration(self) -> dict:
        return {
            "knots": self.knots,
            "low_field_side": True if self.drn == -1 else False
        }

    @classmethod
    def from_configuration(cls, config: dict):
        return cls(
            knots=config["knots"],
            low_field_side=config["low_field_side"]
        )


def build_methods(lib: ModuleType) -> Sequence[callable]:
    def prediction(radius: ndarray, basis: ndarray, drn: int, theta: ndarray) -> ndarray:
        z = (4 * drn / theta[2]) * (radius - theta[0])
        logistic = (1 - theta[1]) / (1 + lib.exp(-z)) + theta[1]
        exp_spline = lib.exp(basis @ theta[3:])
        return logistic * exp_spline

    def jacobian(radius: ndarray, basis: ndarray, drn: int, theta: ndarray):
        exp_spline = lib.exp(basis @ theta[3:])
        z = (4 * drn / theta[2]) * (radius - theta[0])
        L = 1 / (1 + lib.exp(-z))
        q = L * (1 - theta[1])
        dy_df = (1 - L) * exp_spline
        dy_dz = q * dy_df
        y = (q + theta[1]) * exp_spline

        jac = lib.zeros([radius.size, theta.size])
        jac[:, 0] = -dy_dz * (4 * drn / theta[2])
        jac[:, 1] = dy_df
        jac[:, 2] = -dy_dz * z / theta[2]
        jac[:, 2:] = basis * y[:, None]
        return jac

    def prediction_and_jacobian(radius: ndarray, basis: ndarray, drn: int, theta: ndarray):
        exp_spline = lib.exp(basis @ theta[3:])
        z = (4 * drn / theta[2]) * (radius - theta[0])
        L = 1 / (1 + lib.exp(-z))
        q = L * (1 - theta[1])
        dy_df = (1 - L) * exp_spline
        dy_dz = q * dy_df
        y = (q + theta[1]) * exp_spline

        jac = lib.zeros([radius.size, theta.size])
        jac[:, 0] = -dy_dz * (4 * drn / theta[2])
        jac[:, 1] = dy_df
        jac[:, 2] = -dy_dz * z / theta[2]
        jac[:, 2:] = basis * y[:, None]
        return y, jac

    def gradient(radius: ndarray, basis: ndarray, basis_derivs: ndarray, drn: int, theta: ndarray):
        s = lib.exp(basis @ theta[3:])
        z = (4 * drn / theta[2]) * (radius - theta[0])
        L = 1 / (1 + lib.exp(-z))
        q = L * (1 - theta[1])
        return s * ((q + theta[1]) * (basis_derivs @ theta[3:]) + q * (1 - L) * (4 * drn / theta[2]))

    return prediction, jacobian, prediction_and_jacobian, gradient
