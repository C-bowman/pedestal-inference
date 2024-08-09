from functools import partial
from numpy import exp, ndarray, zeros
from pedinf.models import ProfileModel
from pedinf.models.utils import b_spline_basis


class logispline(ProfileModel):
    name = "logispline"

    def __init__(self, knots: ndarray, radius=None, low_field_side=True):
        self.knots = knots
        self.drn = -1 if low_field_side else 1

        self.n_parameters = 3 + self.knots.size
        self.parameters = {
            "pedestal_location": 0,
            "pedestal_height": 1,
            "pedestal_width": 2,
            "basis_weights": slice(3, self.n_parameters),
        }

        if radius is not None:
            self.update_radius(radius)

    def update_radius(self, radius: ndarray):
        self.radius = radius
        self.basis, self.derivs = b_spline_basis(
            self.radius, self.knots, derivatives=True
        )
        self.forward_prediction = partial(self._prediction, self.radius, self.basis)
        self.forward_gradient = partial(self._gradient, self.radius, self.derivs)
        self.forward_jacobian = partial(self._jacobian, self.radius, self.basis)
        self.forward_prediction_and_jacobian = partial(
            self._prediction_and_jacobian, self.radius, self.basis
        )

    def prediction(self, radius: ndarray, theta: ndarray) -> ndarray:
        basis = b_spline_basis(radius, self.knots)
        return self._prediction(radius, basis, theta)

    def gradient(self, radius: ndarray, theta: ndarray) -> ndarray:
        _, derivs = b_spline_basis(radius, self.knots, derivatives=True)
        return self._gradient(radius, derivs, theta)

    def jacobian(self, radius: ndarray, theta: ndarray) -> ndarray:
        basis = b_spline_basis(radius, self.knots)
        return self._jacobian(radius, basis, theta)

    def prediction_and_jacobian(
        self, radius: ndarray, theta: ndarray
    ) -> tuple[ndarray, ndarray]:
        basis = b_spline_basis(radius, self.knots)
        return self._prediction_and_jacobian(radius, basis, theta)

    def _prediction(self, radius: ndarray, basis: ndarray, theta: ndarray) -> ndarray:
        z = (4 * self.drn) * (radius - theta[0]) / theta[2]
        logistic = theta[1] / (1 + exp(-z))
        background = basis @ theta[3:]
        return logistic + background

    def _jacobian(self, radius: ndarray, basis: ndarray, theta: ndarray) -> ndarray:
        z = (4 * self.drn) * (radius - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - L)

        jac = zeros([radius.size, self.n_parameters])
        jac[:, 0] = -q * ((4 * self.drn * theta[1]) / theta[2])
        jac[:, 1] = L
        jac[:, 2] = -q * z * (theta[1] / theta[2])
        jac[:, self.parameters["basis_weights"]] = basis
        return jac

    def _prediction_and_jacobian(
        self, radius: ndarray, basis: ndarray, theta: ndarray
    ) -> tuple[ndarray, ndarray]:
        z = (4 * self.drn) * (radius - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - L)

        prediction = theta[1] * L + basis @ theta[3:]

        jac = zeros([radius.size, self.n_parameters])
        jac[:, 0] = -q * ((4 * self.drn * theta[1]) / theta[2])
        jac[:, 1] = L
        jac[:, 2] = -q * z * (theta[1] / theta[2])
        jac[:, self.parameters["basis_weights"]] = basis
        return prediction, jac

    def _gradient(
        self, radius: ndarray, basis_derivs: ndarray, theta: ndarray
    ) -> ndarray:
        z = (4 * self.drn) * (radius - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - L)
        return q * (4 * self.drn * theta[1] / theta[2]) + basis_derivs @ theta[3:]

    def get_model_configuration(self) -> dict:
        return {
            "knots": self.knots,
            "low_field_side": True if self.drn == -1 else False,
        }

    @classmethod
    def from_configuration(cls, config: dict):
        return cls(knots=config["knots"], low_field_side=config["low_field_side"])
