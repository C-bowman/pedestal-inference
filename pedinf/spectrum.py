from numpy import sqrt, cos, pi, exp
from numpy import ndarray, linspace, zeros
from scipy.interpolate import RectBivariateSpline


def selden(
    Te: ndarray, wavelength: ndarray, theta: ndarray, laser_wavelength=1.064e-6
) -> ndarray:
    """
    Implementation of the 'Selden' equation, which gives a very accurate approximation
    of the relativistic Thomson-scattering spectrum for fusion-relevant temperatures.

    :param Te: \
        The electron temperature in electron-volts.

    :param wavelength: \
        Wavelength values (in meters) at which the scattering spectrum is calculated.

    :param theta: \
        The scattering angle in radians.

    :param laser_wavelength: \
        Wavelength (in meters) of the light being scattered.

    :return: \
        The scattering spectrum. Note that the integral of the Selden equation across
        all scattering wavelengths is equal to the laser wavelength, so the returned
        spectrum must be divided by the laser wavelength to yield a properly normalised
        probability distribution.
    """
    coeff = 3.913902367e-06  # 2 divided by the electron rest-mass-energy in eV
    inv_alpha = coeff * Te
    alpha = 1.0 / inv_alpha

    # Coefficients
    eps = (wavelength - laser_wavelength) / laser_wavelength
    k = 2 * (1 - cos(theta)) * (1 + eps)
    A = (1 + eps) ** 3 * sqrt(k + eps**2)
    B = sqrt(1 + eps**2 / k) - 1
    C = sqrt(alpha / pi) * (1 + inv_alpha * (inv_alpha * (345 / 512) - (15 / 16)))
    spectrum = C / A * exp(-2 * alpha * B)
    return spectrum


def trapezium_weights(x: ndarray) -> ndarray:
    weights = zeros(x.size)
    weights[1:-1] = x[2:] - x[:-2]
    weights[0] = x[1] - x[0]
    weights[-1] = x[-1] - x[-2]
    return weights * 0.5


def calculate_filter_response(
    electron_temperature: ndarray,
    scattering_angle: ndarray,
    wavelength: ndarray,
    transmission: ndarray,
    laser_wavelength=1.064e-6,
) -> ndarray:
    """
    Calculates the integral of the product of the transmission of a given filter and the
    Thomson scattering spectrum for various combinations of electron temperature and
    scattering angle.

    This is useful for generating splines that allow for efficient forward-modelling
    of Thomson-scattering spectrum measurements.

    :param electron_temperature: \
        The electron temperature values (in eV) at which to calculate the filter response.

    :param scattering_angle: \
        The scattering-angle values (in radians) at which to calculate the filter response.

    :param wavelength: \
        The wavelength axis (in meters) corresponding to the given filter transmission values.
        Must be sorted in order of increasing wavelength.

    :param transmission: \
        The transmission values of the filter as a function of wavelength.

    :param laser_wavelength: \
        The wavelength (in meters) of the laser light which being scattered by the
        electrons.

    :return: \
        The filter response values for all pairings of the given electron temperature
        and scattering angle values as a 2D numpy array. If ``n`` electron temperature
        values and ``m`` scattering angle values are given, the returned array will have
        shape ``(n, m)``.
    """
    integration_weights = trapezium_weights(wavelength)
    integration_weights *= transmission / laser_wavelength

    spectrum = selden(
        electron_temperature[None, :, None],
        wavelength[:, None, None],
        scattering_angle[None, None, :],
        laser_wavelength=laser_wavelength,
    )
    return (spectrum * integration_weights[:, None, None]).sum(axis=0)


class SpectralResponse:
    def __init__(
        self, ln_te_axes: ndarray, scattering_angle_axes: ndarray, response: ndarray
    ):
        self.ln_te = ln_te_axes
        self.scattering_angle = scattering_angle_axes
        self.response = response

        # build the splines for all spatial / spectral channels
        n_positions, n_spectra, _, _ = self.response.shape
        self.splines = []
        for j in range(n_spectra):
            self.splines.append(
                [
                    RectBivariateSpline(
                        x=self.ln_te[i, :],
                        y=self.scattering_angle[i, :],
                        z=self.response[i, j, :, :],
                    )
                    for i in range(n_positions)
                ]
            )

    @classmethod
    def calculate_response(
        cls,
        wavelengths: ndarray,
        transmissions: ndarray,
        scattering_angles: ndarray,
        spatial_channels: ndarray,
    ):
        n_temps = 64
        n_angles = 16
        n_spectral_chans = 4
        ln_te = linspace(-3, 10, n_temps)
        te_axis = exp(ln_te)
        delta_angle = linspace(-0.03, 0.03, n_angles)

        response = zeros([spatial_channels.size, n_spectral_chans, n_temps, n_angles])
        ln_te_axes = zeros([spatial_channels.size, n_temps])
        scattering_angle_axes = zeros([spatial_channels.size, n_angles])

        for i, chan in enumerate(spatial_channels):
            ln_te_axes[i, :] = ln_te
            scattering_angle_axes[i, :] = delta_angle + scattering_angles[chan]
            for j in range(n_spectral_chans):
                response[i, j, :, :] = calculate_filter_response(
                    electron_temperature=te_axis,
                    scattering_angle=scattering_angle_axes[i, :],
                    wavelength=wavelengths[chan, j, :],
                    transmission=transmissions[chan, j, :],
                )
        return cls(ln_te_axes, scattering_angle_axes, response)
