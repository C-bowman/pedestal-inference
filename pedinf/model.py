from numpy import exp, log, ndarray, zeros
from scipy.interpolate import RectBivariateSpline


def mtanh(R, theta):
    r"""
    A modified version of the 'mtanh' function which includes an additional parameter
    controlling how rapidly the profile decays at the 'foot' of the pedestal.
    Specifically, the function is:

    .. math::

       f(R, \, \underline{\theta}) = \frac{h(1 - b)(1 - awz)}{(1 + e^{4z})^{k}} + hb,
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
    G = 1 - (a * w) * z
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
    L = L0**-k

    return -(h * (1 - b)) * (G * ((4 * k / w) * exp_4z / L0) + a) * L


def lpm(R, theta):
    R0, h, w, a, b, ln_k = theta
    sigma = 0.25 * w
    z = (R - R0) / sigma
    exp_p1 = 1 + exp(z)
    G = (a * sigma) * (log(exp_p1) - z)
    L = (h - b) * exp_p1 ** -exp(ln_k)
    return (G + L) + b


def lpm_jacobian(R, theta):
    R0, h, w, a, b, ln_k = theta
    k = exp(ln_k)
    z = 4 * (R - R0) / w
    L = 1 / (1 + exp(z))
    S = log(1 + exp(-z))  # think this can be written in terms of L and z
    Lk = L**k

    jac = zeros([R.size, 6])

    df_dz_w = (k * (h - b) / w) * Lk * (1 - L) + (0.25 * a) * L
    jac[:, 0] = -4 * df_dz_w
    jac[:, 1] = Lk
    jac[:, 2] = z * df_dz_w
    jac[:, 3] = (0.25 * w) * S
    jac[:, 4] = 1 - Lk
    jac[:, 5] = (k * (h - b)) * Lk * log(L)
    return jac


class PedestalModel:
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
        L = L0**-k
        GL = G * L
        Q = (exp_4z / L0) * GL

        # fill the jacobian with derivatives of the prediction w.r.t. each parameter
        jac = zeros([self.R.size, 6])
        jac[:, 0] = (h * (1 - b)) * (a * L + (4 * k / w) * Q)
        jac[:, 1] = (1 - b) * GL + b
        jac[:, 2] = (4 * k * h * (1 - b) / w) * Q * z
        jac[:, 3] = -z * (w * h * (1 - b)) * L
        jac[:, 4] = h * (1 - GL)
        jac[:, 5] = -(h * (1 - b)) * GL * log(L0) * k
        return jac


class SpectrometerModel:
    def __init__(
        self,
        response_spline_intensity: ndarray,
        response_spline_ln_te: ndarray,
        response_spline_theta: ndarray,
        inst_func_weights: ndarray,
        inst_func_major_radii: ndarray,
        inst_func_theta: ndarray,
    ):
        self.spline_intensity = response_spline_intensity
        self.spline_ln_te = response_spline_ln_te
        self.spline_theta = response_spline_theta
        self.IF_weights = inst_func_weights
        self.IF_radius = inst_func_major_radii
        self.IF_theta = inst_func_theta

        # make sure the instrument function weights are normalised
        self.IF_weights /= self.IF_weights.sum(axis=1)[:, None]

        self.n_positions = self.spline_intensity.shape[0]
        self.n_spectra = self.spline_intensity.shape[1]

        self.te_slc = slice(0, 6)
        self.ne_slc = slice(6, 12)

        # build the splines for all spatial / spectral channels
        self.splines = []
        for i in range(self.n_positions):
            self.splines.append(
                [
                    RectBivariateSpline(
                        x=self.spline_ln_te[i, :],
                        y=self.spline_theta[i, :],
                        z=self.spline_intensity[i, j, :, :],
                    )
                    for j in range(self.n_spectra)
                ]
            )

    def spectrum(self, Te: ndarray, ne: ndarray) -> ndarray:
        ln_te = log(Te)
        y = zeros([self.n_positions, self.n_spectra])
        coeffs = ne * self.IF_weights
        for i in range(self.n_positions):
            for j in range(self.n_spectra):
                response = self.splines[i][j].ev(ln_te[i, :], self.IF_theta[i, :])
                y[i, j] = (response * coeffs[i, :]).sum()
        return y

    def predictions(self, theta: ndarray) -> ndarray:
        Te = lpm(self.IF_radius, theta[self.te_slc])
        ne = lpm(self.IF_radius, theta[self.ne_slc])
        return self.spectrum(Te, ne).flatten()
