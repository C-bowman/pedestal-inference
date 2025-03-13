from numpy import arange, zeros
from pedinf.diagnostics import SpectrometerModel
from pedinf.models import mtanh
from pedinf.spectrum import SpectralResponse
from tests.data import build_testing_filters, build_testing_instrument
from tests.data import test_channel_scattering_angles


def profile_func(forward_model, test_params):
    for i in range(1000):
        forward_model.predictions_jacobian(test_params)


fit_spatial_channels = arange(107, 130)
inst_data = build_testing_instrument(n_points=20, spatial_channels=fit_spatial_channels)
wavelengths, transmissions = build_testing_filters()


response = SpectralResponse.calculate_response(
    wavelengths=wavelengths,
    transmissions=transmissions,
    scattering_angles=test_channel_scattering_angles,
    spatial_channels=fit_spatial_channels,
    ln_te_range = (-1.5, 8.5),
    n_temps = 128,
    jit_compile = True,
)


forward_model = SpectrometerModel(
    spectral_response=response,
    instrument_function=inst_data,
    profile_model=mtanh()
)


n_params = 2 * forward_model.model.n_parameters
test_params = zeros(n_params)
test_params[forward_model.te_slc] = [1.38, 120., 0.02, 1000., 5.]
test_params[forward_model.ne_slc] = [1.375, 1e19, 0.015, 5e18, 3e17]


# run the function once through to complete jit-compilation
forward_model.predictions(test_params)
forward_model.predictions_jacobian(test_params)
profile_func(forward_model, test_params)

import cProfile
cProfile.run("profile_func(forward_model, test_params)")
