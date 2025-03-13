from numpy import exp, linspace, zeros, ndarray, meshgrid, array
from pedinf.spectrum import SpectralResponse, calculate_filter_response
from data import build_testing_filters, test_channel_scattering_angles


def generate_test_response_data(
    wavelengths: ndarray,
    transmissions: ndarray,
    scattering_angles: ndarray,
    spatial_channels: ndarray,
    ln_te_range=(-3, 10),
    delta_angle_range=(-0.015, 0.015),
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


def test_spectrum_model():
    wavelengths, transmissions = build_testing_filters()
    spatial_channels = array([0, 33, 66, 99, 129])
    n_chans = spatial_channels.size
    # specify knots settings, and also test-points inbetween the knots
    knot_range = (-3., 10.)
    n_knots = 128
    knot_spacing = (knot_range[1] - knot_range[0]) / n_knots
    test_range = (knot_range[0] + 0.5 * knot_spacing, knot_range[1] - 0.5 * knot_spacing)
    n_test_temps = n_knots - 1

    # build the spectrum response model
    response = SpectralResponse.calculate_response(
        wavelengths=wavelengths,
        transmissions=transmissions,
        spatial_channels=spatial_channels,
        scattering_angles=test_channel_scattering_angles,
        ln_te_range=knot_range,
        n_temps=n_knots,
        jit_compile=None
    )

    # generate the target values for spectral response and its gradient
    ln_te_axes, scattering_angle_axes, target_response, target_gradient = generate_test_response_data(
        wavelengths=wavelengths,
        transmissions=transmissions,
        spatial_channels=spatial_channels,
        scattering_angles=test_channel_scattering_angles,
        ln_te_range=test_range,
        n_temps=n_test_temps,
    )

    # combine the variations of ln_te and angle to generate the full testing points
    n_test_points = ln_te_axes.shape[1] * scattering_angle_axes.shape[1]
    ln_te_test_points = zeros([n_chans, n_test_points])
    angle_test_points = zeros([n_chans, n_test_points])

    for i in range(n_chans):
        ln_te_vals, angle_vals = meshgrid(
            ln_te_axes[i, :], scattering_angle_axes[i, :], indexing="ij"
        )
        ln_te_test_points[i, :] = ln_te_vals.flatten()
        angle_test_points[i, :] = angle_vals.flatten()

    # get the predictions of the test values
    spline_gradient, spline_response = response.response_and_grad(
        ln_te=ln_te_test_points,
        scattering_angle=angle_test_points
    )

    # re-shape to match the target values
    spline_response.resize(target_response.shape)
    spline_gradient.resize(target_response.shape)

    response_error = spline_response - target_response
    max_response = array([target_response[:, i, :, :].max() for i in range(4)])
    max_response_errors = array([abs(response_error[:, i, :, :]).max() for i in range(4)])
    assert ((max_response_errors / max_response) < 1e-3).all()

    gradient_error = spline_gradient - target_gradient
    max_gradient = array([target_gradient[:, i, :, :].max() for i in range(4)])
    max_gradient_errors = array([abs(gradient_error[:, i, :, :]).max() for i in range(4)])
    assert ((max_gradient_errors / max_gradient) < 1e-3).all()
