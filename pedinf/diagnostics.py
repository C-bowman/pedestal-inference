from numpy import log, ndarray, zeros
from scipy.interpolate import RectBivariateSpline
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

    def predictions(self, theta: ndarray) -> ndarray:
        Te = self.model.prediction(self.IF_radius, theta[self.te_slc])
        ne = self.model.prediction(self.IF_radius, theta[self.ne_slc])
        return self.spectrum(Te, ne).flatten()