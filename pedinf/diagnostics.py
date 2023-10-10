from dataclasses import dataclass
from numpy import log, ndarray, zeros, einsum
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
        self.model.update_radius(self.instfunc.radius.flatten())

        self.n_positions, self.n_spectra, _, _ = self.response.response.shape
        self.n_predictions = self.n_positions * self.n_spectra
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
        dS_dT *= (coeffs / Te)[:, None, :]
        dS_dn *= self.instfunc.weights[:, None, :]
        return dS_dT, dS_dn

    def predictions(self, theta: ndarray) -> ndarray:
        Te = self.model.forward_prediction(theta[self.te_slc])
        ne = self.model.forward_prediction(theta[self.ne_slc])
        Te.resize(self.instfunc.radius.shape)
        ne.resize(self.instfunc.radius.shape)
        return self.spectrum(Te, ne).flatten()

    def predictions_jacobian(self, theta: ndarray) -> ndarray:
        Te, model_jac_Te = self.model.forward_prediction_and_jacobian(theta[self.te_slc])
        ne, model_jac_ne = self.model.forward_prediction_and_jacobian(theta[self.ne_slc])
        Te.resize(self.instfunc.radius.shape)
        ne.resize(self.instfunc.radius.shape)
        model_jac_Te.resize([*self.instfunc.radius.shape, self.model.n_parameters])
        model_jac_ne.resize([*self.instfunc.radius.shape, self.model.n_parameters])

        dT, dn = self.spectrum_jacobian(Te, ne)

        jac_Te = einsum("iqk, ijq -> ijk", model_jac_Te, dT)
        jac_ne = einsum("iqk, ijq -> ijk", model_jac_ne, dn)

        jac_theta = zeros([self.n_predictions, 2 * self.model.n_parameters])
        jac_theta[:, self.te_slc] = jac_Te.reshape(
            [self.n_predictions, self.model.n_parameters]
        )
        jac_theta[:, self.ne_slc] = jac_ne.reshape(
            [self.n_predictions, self.model.n_parameters]
        )
        return jac_theta
