from abc import ABC, abstractmethod
from typing import Tuple
from numpy import exp, log, ndarray, zeros


class ProfileModel(ABC):
    name: str
    parameters: dict
    n_parameters: int

    @staticmethod
    @abstractmethod
    def prediction(R: ndarray, theta: ndarray) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def gradient(R: ndarray, theta: ndarray) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def jacobian(R: ndarray, theta: ndarray) -> ndarray:
        pass

    @staticmethod
    @abstractmethod
    def prediction_and_jacobian(R: ndarray, theta: ndarray) -> Tuple[ndarray]:
        pass


class mtanh(ProfileModel):
    r"""
    The standard 'mtanh' function used for pedestal fitting. Specifically,
    the function is:

    .. math::

       \mathrm{mtanh}(R, \, \underline{\theta}) = L(z) \left(h + \frac{awz}{4} \right) + b,
       \quad \quad L(x) = \frac{1}{1 + e^{-x}} \quad \quad z = -4 \frac{R - R_0}{w}.

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
        "R0 (pedestal location)": 0,
        "h (pedestal height)": 1,
        "w (pedestal width)": 2,
        "a (pedestal top gradient)": 3,
        "b (background level)": 4,
    }

    @staticmethod
    def prediction(R: ndarray, theta: ndarray) -> ndarray:
        r"""
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
    def prediction_and_jacobian(R: ndarray, theta: ndarray) -> Tuple[ndarray]:
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


class lpm(ProfileModel):
    r"""
    A modified version of the 'mtanh' function which includes an additional parameter
    controlling how rapidly the profile decays at the 'foot' of the pedestal.
    Specifically, the function is:

    .. math::

       \mathrm{lpm}(R, \, \underline{\theta}) = h\,L^{k}(z) + \frac{aw}{4}S(z) + b,
       \quad \quad L(x) = \frac{1}{1 + e^{-x}} \quad \quad z = -4 \frac{R - R_0}{w}.

    and

    .. math::

       S(x) = \int_{-\infty}^{x} L(x')\,\mathrm{d}x' = \ln{(1 + e^x)}

    The model parameter vector :math:`\underline{\theta}` has the following order:

    .. math::

       \underline{\theta} = \left[ \,  R_0, \, h, \, w, \, a, \, b, \, \ln{k} \, \right],

    where

     - :math:`R_0` is the radial location of the pedestal.
     - :math:`h` is the pedestal height.
     - :math:`w` is the pedestal width.
     - :math:`a` is the profile gradient beyond the pedestal top.
     - :math:`b` is the background level.
     - :math:`\ln{k}` is a shaping parameter which affects how the profile decays.
    """
    name = "lpm"
    n_parameters = 6
    parameters = {
        "R0 (pedestal location)": 0,
        "h (pedestal height)": 1,
        "w (pedestal width)": 2,
        "a (pedestal top gradient)": 3,
        "b (background level)": 4,
        "ln_k (logistic shape parameter)": 5,
    }

    @staticmethod
    def prediction(R: ndarray, theta: ndarray) -> ndarray:
        r"""
        Calculates the prediction of the ``lpm`` model.
        See the documentation for ``lpm`` for details of the model itself.

        :param R: \
            Radius values at which the gradient is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The predicted profile at the given radius values.
        """
        R0, h, w, a, b, ln_k = theta
        sigma = 0.25 * w
        z = (R - R0) / sigma
        exp_p1 = 1 + exp(z)
        G = (a * sigma) * (log(exp_p1) - z)
        L = (h - b) * exp_p1 ** -exp(ln_k)
        return (G + L) + b

    @staticmethod
    def gradient(R: ndarray, theta: ndarray) -> ndarray:
        """
        Calculates the gradient (w.r.t. major radius) of the ``lpm`` model.
        See the documentation for ``lpm`` for details of the model itself.

        :param R: \
            Radius values at which the gradient is evaluated.

        :param theta: \
            The model parameters as an array.

        :return: \
            The predicted gradient profile at the given radius values.
        """
        R0, h, w, a, b, ln_k = theta
        k = exp(ln_k)
        z = 4 * (R - R0) / w
        L = 1 / (1 + exp(z))
        return -(4 * (h - b) * k / w) * (1 - L) * L**k - a * L

    @staticmethod
    def jacobian(R: ndarray, theta: ndarray) -> ndarray:
        R0, h, w, a, b, ln_k = theta
        k = exp(ln_k)
        z = 4 * (R - R0) / w
        L = 1 / (1 + exp(z))
        ln_L = log(L)
        S = -ln_L - z
        Lk = L**k

        jac = zeros([R.size, 6])
        df_dz = (k * (h - b)) * Lk * (1 - L) + (0.25 * a * w) * L
        jac[:, 0] = (4 / w) * df_dz
        jac[:, 1] = Lk
        jac[:, 2] = (z / w) * df_dz + (0.25 * a) * S
        jac[:, 3] = (0.25 * w) * S
        jac[:, 4] = 1 - Lk
        jac[:, 5] = (k * (h - b)) * Lk * ln_L
        return jac

    @staticmethod
    def prediction_and_jacobian(R: ndarray, theta: ndarray) -> Tuple[ndarray]:
        R0, h, w, a, b, ln_k = theta
        k = exp(ln_k)
        z = 4 * (R - R0) / w
        L = 1 / (1 + exp(z))
        ln_L = log(L)
        S = -ln_L - z
        Lk = L**k

        prediction = Lk * (h - b) + (0.25 * a * w) * S + b
        jac = zeros([R.size, 6])
        df_dz = (k * (h - b)) * Lk * (1 - L) + (0.25 * a * w) * L
        jac[:, 0] = (4 / w) * df_dz
        jac[:, 1] = Lk
        jac[:, 2] = (z / w) * df_dz + (0.25 * a) * S
        jac[:, 3] = (0.25 * w) * S
        jac[:, 4] = 1 - Lk
        jac[:, 5] = (k * (h - b)) * Lk * ln_L
        return prediction, jac
