from numpy import exp, log, zeros


def mtanh(R, theta):
    """
    A modified version of the standard 'mtanh' function which includes
    an additional parameter controlling how rapidly the profile decays
    at the 'foot' of the pedestal.

    :param R: \
        Radius values at which the profile is evaluated.

    :param theta: \
        The model parameters as an array or list.

    :return: \
        The predicted profile at the given radius values.
    """
    R0, h, w, a, b, ln_k = theta
    z = (R - R0) / w
    G = 1 - a * z
    L = (1 + exp(4 * z)) ** -exp(ln_k)
    return (h * (1 - b)) * (G * L) + h * b


def mtanh_gradient(R, theta):
    """
    Calculates the gradient (w.r.t. major radius) of a modified
    version of the standard 'mtanh' function which includes an additional
    parameter controlling how rapidly the profile decays at the 'foot' of
    the pedestal.

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
    G = 1 - a * z
    exp_4z = exp(4 * z)
    L0 = 1 + exp_4z
    L = L0 ** -k

    # the derivative of the prediction w.r.t. 'z'
    df_dz = -(h * (1 - b)) * (G * ((4 * k) * exp_4z / L0) + a) * L
    return df_dz / w


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
        G = 1 - a * z
        exp_4z = exp(4 * z)
        L0 = 1 + exp_4z
        L = L0 ** -k
        GL = G * L

        # the derivative of the prediction w.r.t. 'z'
        df_dz = -(h*(1 - b)) * (GL*((4*k) * exp_4z / L0) + a*L)

        # fill the jacobian with derivatives of the prediction w.r.t. each parameter
        jac = zeros([self.R.size, 6])
        jac[:, 0] = -df_dz / w
        jac[:, 1] = (1 - b)*GL + b
        jac[:, 2] = jac[:, 0] * z
        jac[:, 3] = -z * (h*(1 - b)) * L
        jac[:, 4] = h*(1 - GL)
        jac[:, 5] = -(h*(1 - b)) * GL * log(L0) * k
        return jac
