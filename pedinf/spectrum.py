from numpy import sqrt, cos, pi, exp
from numpy import ndarray, zeros


def selden(Te, wavelength, theta, laser_wavelength=1.064e-6):
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


def trapezium_weights(x):
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

    integration_weights = trapezium_weights(wavelength)
    integration_weights *= transmission / laser_wavelength

    spectrum = selden(
        electron_temperature[None, :, None],
        wavelength[:, None, None],
        scattering_angle[None, None, :],
        laser_wavelength=laser_wavelength,
    )
    return (spectrum * integration_weights[:, None, None]).sum(axis=0)
