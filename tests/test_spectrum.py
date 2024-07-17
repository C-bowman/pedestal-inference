from numpy import exp, linspace, zeros, arange, ndarray
from pedinf.spectrum import SpectralResponse1D, calculate_filter_response


def calculate_response(
    wavelengths: ndarray,
    transmissions: ndarray,
    scattering_angles: ndarray,
    spatial_channels: ndarray,
    ln_te_range=(-3, 10),
    delta_angle_range=(-0.03, 0.03),
    n_temps=64,
    n_angles=16,
):
    n_spectral_chans = 4
    ln_te = linspace(*ln_te_range, n_temps)
    te_axis = exp(ln_te)
    delta_angle = linspace(*delta_angle_range, n_angles)

    response = zeros([spatial_channels.size, n_spectral_chans, n_temps, n_angles])
    gradient = zeros([spatial_channels.size, n_spectral_chans, n_temps, n_angles])
    ln_te_axes = zeros([spatial_channels.size, n_temps])
    scattering_angle_axes = zeros([spatial_channels.size, n_angles])
    dte = 1e-4

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

            f1 = calculate_filter_response(
                electron_temperature=exp(ln_te - dte),
                scattering_angle=scattering_angle_axes[i, :],
                wavelength=wavelengths[chan, j, :],
                transmission=transmissions[chan, j, :],
            )

            f2 = calculate_filter_response(
                electron_temperature=exp(ln_te + dte),
                scattering_angle=scattering_angle_axes[i, :],
                wavelength=wavelengths[chan, j, :],
                transmission=transmissions[chan, j, :],
            )

            gradient[i, j, :, :] = 0.5 * (f2 - f1) / dte
    return ln_te_axes, scattering_angle_axes, response, gradient


def superguass(x, c, w, n=4):
    z = (x - c) / w
    return exp(-0.5 * z**n)


def generate_testing_filters():
    # parameters for a set of testing polychromator filters
    # generated using super-gaussians
    wavelength_starts = [1.053e-06, 1.035e-06, 9.8e-07, 8.1e-07]
    wavelength_ends = [1.064e-06, 1.06e-06, 1.0462e-06, 1.0075e-06]
    amplitudes = [1.0, 1.14, 1.43, 1.6]
    widths = [2.55e-9, 7e-9, 2.1e-8, 7.1e-8]
    centres = [1.058e-6, 1.0477e-6, 1.018e-6, 9.19e-7]
    exponents = [4, 8, 10, 12]

    # build the transmission data
    wavelengths = zeros([130, 4, 128])
    transmissions = zeros([130, 4, 128])
    for i in range(4):
        wavelengths[:, i, :] = linspace(wavelength_starts[i], wavelength_ends[i], 128)[None, :]
        transmissions[:, i, :] = amplitudes[i] * superguass(
            wavelengths[:, i, :], c=centres[i], w=widths[i], n=exponents[i]
        )[None, :]

    spatial_channels = arange(0, 130)
    # generate testing scattering angles from a quadratric
    coeffs = [7.85941e-07, -6.43046e-03, 2.1297]
    scattering_angles = coeffs[2] + spatial_channels * coeffs[1] + spatial_channels ** 2 * coeffs[0]
    return wavelengths, transmissions, scattering_angles, spatial_channels


wavelengths, transmissions, scattering_angles, spatial_channels = generate_testing_filters()
response = SpectralResponse1D.calculate_response(
    wavelengths=wavelengths,
    transmissions=transmissions,
    spatial_channels=spatial_channels,
    scattering_angles=scattering_angles
)
