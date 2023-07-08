from dataclasses import dataclass
from numpy import log, ndarray, zeros
from pedinf.models import ProfileModel
from pedinf.spectrum import SpectralResponse


@dataclass
class InstrumentFunction:
    radius: ndarray
    scattering_angle: ndarray
    weights: ndarray

    def __post_init__(self):
        # make sure the instrument function weights are normalised
        self.weights /= self.weights.sum(axis=1)[:, None]


class SpectrometerModel:
    def __init__(
        self,
        spectral_response: SpectralResponse,
        instrument_function: InstrumentFunction,
        profile_model: ProfileModel,
    ):
        self.response = spectral_response
        self.instfunc = instrument_function
        self.model = profile_model

        self.n_positions, self.n_spectra, _, _ = self.response.response.shape
        self.n_weights = self.instfunc.weights.shape[1]
        self.spectrum_shape = (self.n_positions, self.n_spectra, self.n_weights)
        self.te_slc = slice(0, self.model.n_parameters)
        self.ne_slc = slice(self.model.n_parameters, 2 * self.model.n_parameters)

    def spectrum(self, Te: ndarray, ne: ndarray) -> ndarray:
        ln_te = log(Te)
        y = zeros(self.spectrum_shape)
        coeffs = ne * self.instfunc.weights
        for j in range(self.n_spectra):
            splines = self.response.splines[j]
            for i in range(self.n_positions):
                y[i, j, :] = splines[i].ev(
                    ln_te[i, :], self.instfunc.scattering_angle[i, :]
                )
        y *= coeffs[:, None, :]
        return y.sum(axis=2)

    def spectrum_jacobian(self, Te: ndarray, ne: ndarray):
        ln_te = log(Te)
        dS_dT = zeros(self.spectrum_shape)
        dS_dn = zeros(self.spectrum_shape)
        coeffs = ne * self.instfunc.weights
        for j in range(self.n_spectra):
            splines = self.response.splines[j]
            for i in range(self.n_positions):
                dS_dn[i, j, :] = splines[i].ev(
                    ln_te[i, :], self.instfunc.scattering_angle[i, :]
                )
                dS_dT[i, j, :] = splines[i].ev(
                    ln_te[i, :], self.instfunc.scattering_angle[i, :], dx=1
                )
        dS_dT *= coeffs[:, None, :]
        dS_dT /= Te[:, None, :]
        dS_dn *= self.instfunc.weights[:, None, :]
        return dS_dT, dS_dn

    def predictions(self, theta: ndarray) -> ndarray:
        Te = self.model.prediction(self.instfunc.radius, theta[self.te_slc])
        ne = self.model.prediction(self.instfunc.radius, theta[self.ne_slc])
        return self.spectrum(Te, ne).flatten()

    def jacobian(self, theta: ndarray):
        Te, model_Te_jac = self.model.prediction_and_jacobian(self.instfunc.radius.flatten(), theta[self.te_slc])
        ne, model_ne_jac = self.model.prediction_and_jacobian(self.instfunc.radius.flatten(), theta[self.ne_slc])
        Te.resize(self.instfunc.radius.shape)
        ne.resize(self.instfunc.radius.shape)
        model_Te_jac.resize([*self.instfunc.radius.shape, self.model.n_parameters])
        model_ne_jac.resize([*self.instfunc.radius.shape, self.model.n_parameters])

        dT, dn = self.spectrum_jacobian(Te, ne)
        print(dT.shape, model_Te_jac.shape)
        Jac_Te = zeros([self.n_positions, self.n_spectra, self.model.n_parameters])
        Jac_ne = zeros([self.n_positions, self.n_spectra, self.model.n_parameters])
        for i in range(self.n_positions):
            for j in range(self.n_spectra):
                Jac_Te[i, j, :] = model_Te_jac[i, :, :].T @ dT[i, j, :]
                Jac_ne[i, j, :] = model_ne_jac[i, :, :].T @ dn[i, j, :]
        Jac_Te.resize([self.n_positions * self.n_spectra, 5])
        Jac_ne.resize([self.n_positions * self.n_spectra, 5])
        return Jac_Te, Jac_ne