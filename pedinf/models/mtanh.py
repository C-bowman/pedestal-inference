from typing import Tuple
from numpy import ndarray, exp, zeros
from pedinf.models.base import ProfileModel


class mtanh(ProfileModel):
    r"""
    The standard 'mtanh' function used for pedestal fitting. Specifically,
    the function is:

    .. math::

       \mathrm{mtanh}(R, \, \underline{\theta}) = L(z) \left(h - b + \frac{awz}{4} \right) + b

    where

    .. math::

       z = -4 \frac{R - R_0}{w}, \quad \quad L(x) = \frac{1}{1 + e^{-x}}.

    The model parameter vector :math:`\underline{\theta}` has the following order:

    .. math::

       \underline{\theta} = \left[ \,  R_0, \, h, \, w, \, a, \, b \, \right],

    where

     - :math:`R_0` is the radial location of the pedestal.
     - :math:`h` is the pedestal height.
     - :math:`w` is the pedestal width.
     - :math:`a` is the profile gradient beyond the pedestal top.
     - :math:`b` is the background level.
    """

    name = "mtanh"
    n_parameters = 5
    parameters = {
        "pedestal_location": 0,
        "pedestal_height": 1,
        "pedestal_width": 2,
        "pedestal_top_gradient": 3,
        "background_level": 4,
    }

    @staticmethod
    def prediction(R: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the prediction of the ``mtanh`` model.
        See the documentation for ``mtanh`` for details of the model itself.

        :param R: \
            Radius values at which the gradient is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The predicted profile at the given radius values.
        """
        R0, h, w, a, b = theta
        sigma = 0.25 * w
        z = (R0 - R) / sigma
        G = h - b + (a * sigma) * z
        iL = 1 + exp(-z)
        return (G / iL) + b

    @staticmethod
    def gradient(R: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the gradient (w.r.t. major radius) of the ``mtanh`` function.
        See the documentation for ``mtanh`` for details of the function itself.

        :param R: \
            Radius values at which the gradient is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The predicted gradient profile at the given radius values.
        """
        R0, h, w, a, b = theta
        sigma = 0.25 * w
        z = (R0 - R) / sigma
        L = 1 / (1 + exp(-z))
        c = (h - b) / sigma
        return -L * ((1 - L) * (c + a * z) + a)

    @staticmethod
    def jacobian(R: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the jacobian of the ``mtanh`` model. The jacobian is a matrix where
        element :math:`i, j` is the derivative of the model prediction at the
        :math:`i`'th radial position with respect to the :math:`j`'th model parameter.
        See the documentation for ``mtanh`` for details of the model itself.

        :param R: \
            Radius values at which the gradient is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The jacobian matrix for the given radius values.
        """
        R0, h, w, a, b = theta
        sigma = 0.25 * w
        z = (R0 - R) / sigma
        L = 1 / (1 + exp(-z))

        jac = zeros([R.size, 5])
        c = (h - b) / sigma
        q = L * (1 - L) * (c + a * z)
        jac[:, 0] = q + L * a
        jac[:, 1] = L
        jac[:, 2] = -0.25 * q * z
        jac[:, 3] = (sigma * z) * L
        jac[:, 4] = 1 - L
        return jac

    @staticmethod
    def prediction_and_jacobian(R: ndarray, theta: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Calculates the prediction and the jacobian of the ``mtanh`` model. The jacobian
        is a matrix where element :math:`i, j` is the derivative of the model prediction
        at the :math:`i`'th radial position with respect to the :math:`j`'th model parameter.
        See the documentation for ``mtanh`` for details of the model itself.

        :param R: \
            Radius values at which the gradient is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The model prediction and the jacobian matrix for the given radius values.
        """
        R0, h, w, a, b = theta
        sigma = 0.25 * w
        z = (R0 - R) / sigma
        L = 1 / (1 + exp(-z))
        G = (h - b) + (a * sigma) * z
        prediction = L * G + b

        jac = zeros([R.size, 5])
        q = L * (1 - L) * G / sigma
        jac[:, 0] = q + L * a
        jac[:, 1] = L
        jac[:, 2] = -0.25 * q * z
        jac[:, 3] = (sigma * z) * L
        jac[:, 4] = 1 - L
        return prediction, jac
