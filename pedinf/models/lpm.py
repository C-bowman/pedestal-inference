from functools import partial
from numpy import ndarray, log, exp, zeros
from pedinf.models.base import ProfileModel


class lpm(ProfileModel):
    r"""
    A modified version of the 'mtanh' function which includes an additional parameter
    controlling how rapidly the profile decays at the 'foot' of the pedestal.
    Specifically, the function is:

    .. math::

       \mathrm{lpm}(R, \, \underline{\theta}) = (h - b)\,L^{k}(z + \ln{k}) + \frac{aw}{4}S(z) + b

    where

    .. math::

       z = -4 \frac{R - R_0}{w}, \quad \quad L(x) = \frac{1}{1 + e^{-x}}, \quad \quad
       S(x) = \int_{-\infty}^{x} L(x')\,\mathrm{d}x' = \ln{(1 + e^x)}

    The model parameter vector :math:`\underline{\theta}` has the following order:

    .. math::

       \underline{\theta} = \left[ \,  R_0, \, h, \, w, \, a, \, b, \, k \, \right],

    where

     - :math:`R_0` is the radial location of the pedestal.
     - :math:`h` is the pedestal height.
     - :math:`w` is the pedestal width.
     - :math:`a` is the profile gradient beyond the pedestal top.
     - :math:`b` is the background level.
     - :math:`k` is a shaping parameter which affects how the profile decays.
    """

    name = "lpm"
    n_parameters = 6
    parameters = {
        "pedestal_location": 0,
        "pedestal_height": 1,
        "pedestal_width": 2,
        "pedestal_top_gradient": 3,
        "background_level": 4,
        "logistic_shape_parameter": 5,
    }

    def __init__(self, radius=None, low_field_side=True):
        self.drn = -1 if low_field_side else 1
        if radius is not None:
            self.update_radius(radius)

    def update_radius(self, radius: ndarray):
        self.radius = radius
        self.forward_prediction = partial(self.prediction, self.radius)
        self.forward_gradient = partial(self.gradient, self.radius)
        self.forward_jacobian = partial(self.jacobian, self.radius)
        self.forward_prediction_and_jacobian = partial(
            self.prediction_and_jacobian, self.radius
        )

    def prediction(self, radius: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the prediction of the ``lpm`` model.
        See the documentation for ``lpm`` for details of the model itself.

        :param radius: \
            Radius values at which the prediction is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The predicted profile at the given radius values.
        """
        R0, h, w, a, b, k = theta
        sigma = 0.25 * w
        ln_k = log(k)
        z = (radius - R0) * (self.drn / sigma)
        iL = 1 + exp(-z - ln_k)
        G = (a * sigma) * log(1 + exp(z))
        L = (h - b) * iL**-k
        return G + L + b

    def gradient(self, radius: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the gradient (w.r.t. major radius) of the ``lpm`` model.
        See the documentation for ``lpm`` for details of the model itself.

        :param radius: \
            Radius values at which the gradient is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The predicted gradient profile at the given radius values.
        """
        R0, h, w, a, b, k = theta
        sigma = 0.25 * w
        z = (radius - R0) * (self.drn / sigma)
        exp_z = exp(-z)
        L = k / (k + exp_z)
        return (((h - b) * k / sigma) * (1 - L) * L**k + a / (1 + exp_z)) * self.drn

    def jacobian(self, radius: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the jacobian of the ``lpm`` model. The jacobian is a matrix where
        element :math:`i, j` is the derivative of the model prediction at the
        :math:`i`'th radial position with respect to the :math:`j`'th model parameter.
        See the documentation for ``lpm`` for details of the model itself.

        :param radius: \
            Radius values at which the jacobian is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The jacobian matrix for the given radius values.
        """
        R0, h, w, a, b, k = theta
        sigma = 0.25 * w
        z = (radius - R0) * (self.drn / sigma)
        exp_z = exp(-z)
        L = k / (k + exp_z)
        ln_L = log(L)
        S = log(1 + exp_z) + z
        Lk = L**k

        jac = zeros([radius.size, 6])
        df_dz = (k * (h - b)) * Lk * (1 - L) + (a * sigma) / (1 + exp_z)
        jac[:, 0] = -df_dz * (self.drn / sigma)
        jac[:, 1] = Lk
        jac[:, 2] = -(z / w) * df_dz + (0.25 * a) * S
        jac[:, 3] = sigma * S
        jac[:, 4] = 1 - Lk
        jac[:, 5] = (h - b) * Lk * (1 + ln_L - L)
        return jac

    def prediction_and_jacobian(
        self, radius: ndarray, theta: ndarray
    ) -> tuple[ndarray, ndarray]:
        """
        Calculates the prediction and the jacobian of the ``lpm`` model. The jacobian
        is a matrix where element :math:`i, j` is the derivative of the model prediction
        at the :math:`i`'th radial position with respect to the :math:`j`'th model parameter.
        See the documentation for ``lpm`` for details of the model itself.

        :param radius: \
            Radius values at which the prediction and jacobian are evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The model prediction and the jacobian matrix for the given radius values.
        """
        R0, h, w, a, b, k = theta
        sigma = 0.25 * w
        z = (radius - R0) * (self.drn / sigma)
        exp_z = exp(-z)
        L = k / (k + exp_z)
        ln_L = log(L)
        S = log(1 + exp_z) + z
        Lk = L**k

        prediction = Lk * (h - b) + (a * sigma) * S + b
        jac = zeros([radius.size, 6])
        df_dz = (k * (h - b)) * Lk * (1 - L) + (a * sigma) / (1 + exp_z)
        jac[:, 0] = -df_dz * (self.drn / sigma)
        jac[:, 1] = Lk
        jac[:, 2] = -(z / w) * df_dz + (0.25 * a) * S
        jac[:, 3] = sigma * S
        jac[:, 4] = 1 - Lk
        jac[:, 5] = (h - b) * Lk * (1 + ln_L - L)
        return prediction, jac

    def get_model_configuration(self) -> dict:
        return {"low_field_side": True if self.drn == -1 else False}

    @classmethod
    def from_configuration(cls, config: dict):
        return cls(low_field_side=config["low_field_side"])
