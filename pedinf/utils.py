from numpy import exp, linspace, zeros, tile
from pedinf.diagnostics import InstrumentFunction


def scattering_angle(R):
    return 2.27 - 0.65 * R


test_channel_radius = linspace(0.21, 1.50, 130)
test_channel_scattering_angles = scattering_angle(test_channel_radius)


def superguass(x, c, w, n=4):
    z = (x - c) / w
    return exp(-0.5 * z**n)


def logistic(x):
    return 1 / (1 + exp(-x))


def logistic_product(x, c, w, sigma):
    z1 = (x - (c - w)) / sigma
    z2 = (x - (c + w)) / sigma
    return logistic(z1) * (1 - logistic(z2))


def build_testing_instrument(n_points: int = 20, spatial_channels=None):
    R0 = test_channel_radius if spatial_channels is None else test_channel_radius[spatial_channels]
    dR = linspace(-0.0075, 0.0075, n_points)
    R = R0[:, None] + dR[None, :]
    angles = scattering_angle(R)

    y = logistic_product(dR, c=0.0, w=0.005, sigma=4e-4)
    weights = tile(y, (R0.size, 1))

    return InstrumentFunction(
        radius=R,
        scattering_angle=angles,
        weights=weights
    )


def build_testing_filters():
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

    return wavelengths, transmissions