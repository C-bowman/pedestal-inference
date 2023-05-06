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

        self.n_positions, self.n_spectra = self.response.response.shape
        self.n_weights = self.instfunc.weights.shape[1]

        self.te_slc = slice(0, self.model.n_parameters)
        self.ne_slc = slice(self.model.n_parameters, 2 * self.model.n_parameters)

    def spectrum(self, Te: ndarray, ne: ndarray) -> ndarray:
        ln_te = log(Te)
        y = zeros([self.n_positions, self.n_spectra])
        coeffs = ne * self.instfunc.weights
        for i in range(self.n_positions):
            for j in range(self.n_spectra):
                response = self.response.splines[i][j].ev(
                    ln_te[i, :], self.instfunc.scattering_angle[i, :]
                )
                y[i, j] = (response * coeffs[i, :]).sum()
        return y

    def predictions(self, theta: ndarray) -> ndarray:
        Te = self.model.prediction(self.instfunc.radius, theta[self.te_slc])
        ne = self.model.prediction(self.instfunc.radius, theta[self.ne_slc])
        return self.spectrum(Te, ne).flatten()
