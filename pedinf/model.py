from numpy import exp, log, zeros


def mtanh(R, theta):
    r"""
    A modified version of the 'mtanh' function which includes an additional parameter
    controlling how rapidly the profile decays at the 'foot' of the pedestal.
    Specifically, the function is:

    .. math::

       f(R, \, \underline{\theta}) = \frac{h(1 - b)(1 - az)}{(1 + e^{4z})^{k}} + hb,
       \quad \quad z = \frac{R - R_0}{w}.

    The model parameter vector :math:`\underline{\theta}` has the following order:

    .. math::

       \underline{\theta} = \left[ \,  R_0, \, h, \, w, \, a, \, b, \, \ln{k} \, \right],

    where

     - :math:`R_0` is the radial location of the pedestal.
     - :math:`h` is the pedestal height.
     - :math:`w` is the pedestal width.
     - :math:`a` controls the profile gradient at the pedestal top.
     - :math:`b` sets the background level as a fraction of the pedestal height.
     - :math:`\ln{k}` is a shaping parameter which affects how the profile decays.

    :param R: Radius values at which the profile is evaluated.
    :param theta: The model parameters as an array or list.
    :return: The predicted profile at the given radius values.
    """
    R0, h, w, a, b, ln_k = theta
    z = (R - R0) / w
    G = 1 - (a*w) * z
    L = (1 + exp(4 * z)) ** -exp(ln_k)
    return (h * (1 - b)) * (G * L) + h * b


def mtanh_gradient(R, theta):
    """
    Calculates the gradient (w.r.t. major radius) of the ``mtanh`` function.
    See the documentation for ``mtanh`` for details of the function itself.

    :param R: \
        Radius values at which the gradient is evaluated.

    :param theta: \
        The model parameters as an array or list.

    :return: \
        The predicted gradient profile at the given radius values.
    """
    R0, h, w, a, b, ln_k = theta

    # pre-calculate some quantities for optimisation
    k = exp(ln_k)
    z = (R - R0) / w
    G = 1 - (a * w) * z
    exp_4z = exp(4 * z)
    L0 = 1 + exp_4z
    L = L0 ** -k

    return -(h * (1 - b)) * (G * ((4 * k / w) * exp_4z / L0) + a) * L


class PedestalModel(object):
    def __init__(self, R):
        self.R = R

    def prediction(self, theta):
        return mtanh(self.R, theta)

    def jacobian(self, theta):
        R0, h, w, a, b, ln_k = theta

        # pre-calculate some quantities for optimisation
        k = exp(ln_k)
        z = (self.R - R0) / w
        G = 1 - (a * w) * z
        exp_4z = exp(4 * z)
        L0 = 1 + exp_4z
        L = L0 ** -k
        GL = G * L
        Q = (exp_4z/L0)*GL

        # fill the jacobian with derivatives of the prediction w.r.t. each parameter
        jac = zeros([self.R.size, 6])
        jac[:, 0] = (h*(1 - b))*(a*L + (4*k/w)*Q)
        jac[:, 1] = (1 - b)*GL + b
        jac[:, 2] = (4*k*h*(1 - b)/w) * Q * z
        jac[:, 3] = -z * (w*h*(1 - b)) * L
        jac[:, 4] = h*(1 - GL)
        jac[:, 5] = -(h*(1 - b)) * GL * log(L0) * k
        return jac
