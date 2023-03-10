from numpy import log, ndarray, zeros
from scipy.interpolate import RectBivariateSpline
from itertools import product
from pedinf.models import ProfileModel


class SpectrometerModel:
    def __init__(
        self,
        response_spline_intensity: ndarray,
        response_spline_ln_te: ndarray,
        response_spline_theta: ndarray,
        inst_func_weights: ndarray,
        inst_func_major_radii: ndarray,
        inst_func_theta: ndarray,
        profile_model: ProfileModel
    ):
        self.spline_intensity = response_spline_intensity
        self.spline_ln_te = response_spline_ln_te
        self.spline_theta = response_spline_theta
        self.IF_weights = inst_func_weights
        self.IF_radius = inst_func_major_radii
        self.IF_theta = inst_func_theta
        self.model = profile_model

        # make sure the instrument function weights are normalised
        self.IF_weights /= self.IF_weights.sum(axis=1)[:, None]

        self.n_positions = self.spline_intensity.shape[0]
        self.n_spectra = self.spline_intensity.shape[1]
        self.n_weights = self.IF_weights.shape[1]

        self.te_slc = slice(0, self.model.n_parameters)
        self.ne_slc = slice(self.model.n_parameters, 2*self.model.n_parameters)

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

    def spectrum_jacobian(self, Te: ndarray, ne: ndarray):
        ln_te = log(Te)
        te_jac = zeros([self.n_positions, self.n_spectra, self.n_positions, self.n_weights])
        ne_jac = zeros([self.n_positions, self.n_spectra, self.n_positions, self.n_weights])
        coeffs = ne * Te * self.IF_weights
        for i in range(self.n_positions):
            for j in range(self.n_spectra):
                ne_jac[i, j, i, :] = self.IF_weights[i, :] * self.splines[i][j].ev(ln_te[i, :], self.IF_theta[i, :])
                te_jac[i, j, i, :] = coeffs[i, :] * self.splines[i][j].ev(ln_te[i, :], self.IF_theta[i, :], dx=1)
        te_jac.resize([self.n_positions*self.n_spectra, self.n_positions*self.n_weights])
        ne_jac.resize([self.n_positions*self.n_spectra, self.n_positions*self.n_weights])
        return te_jac, ne_jac

    def spectrum_jacobian_alt(self, Te: ndarray, ne: ndarray):
        ln_te = log(Te)
        te_jac = zeros([self.n_positions*self.n_spectra, self.n_positions*self.n_weights])
        ne_jac = zeros([self.n_positions*self.n_spectra, self.n_positions*self.n_weights])
        coeffs = ne * Te * self.IF_weights
        itr1 = enumerate(product(range(self.n_positions, self.n_spectra)))
        itr2 = enumerate(product(range(self.n_positions, self.n_weights)))
        for a, (i, j) in itr1:
            for b, (k, l) in itr2:
                if i == k:
                    ne_jac[a, b] = self.IF_weights[i, l] * self.splines[i][j].ev(ln_te[i, l], self.IF_theta[i, l])
                    te_jac[a, b] = coeffs[i, l] * self.splines[i][j].ev(ln_te[i, l], self.IF_theta[i, l], dx=1)
        return te_jac, ne_jac

    def predictions(self, theta: ndarray) -> ndarray:
        Te = self.model.prediction(self.IF_radius, theta[self.te_slc])
        ne = self.model.prediction(self.IF_radius, theta[self.ne_slc])
        return self.spectrum(Te, ne).flatten()

    def predictions_jacobian(self, theta: ndarray) -> ndarray:
        Te, lpm_te_jac = self.model.prediction_and_jacobian(self.IF_radius.flatten(), theta[self.te_slc])
        ne, lpm_ne_jac = self.model.prediction_and_jacobian(self.IF_radius.flatten(), theta[self.ne_slc])
        spec_te_jac, spec_ne_jac = self.spectrum_jacobian(
            Te.reshape([self.n_positions, self.n_weights]),
            ne.reshape([self.n_positions, self.n_weights])
        )

        spec_te_jac_alt, spec_ne_jac_alt = self.spectrum_jacobian_alt(
            Te.reshape([self.n_positions, self.n_weights]),
            ne.reshape([self.n_positions, self.n_weights])
        )

        jac = zeros([self.n_positions * self.n_spectra, 12])
        jac[:, :6] = spec_te_jac_alt @ lpm_te_jac
        jac[:, 6:] = spec_ne_jac_alt @ lpm_ne_jac
        return jac