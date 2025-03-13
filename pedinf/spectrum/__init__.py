from numpy import array, sqrt, cos, pi, exp, log, stack
from numpy import ndarray, linspace, zeros, float32
from scipy.interpolate import RectBivariateSpline
from pedinf.spline import cubic_spline_coefficients


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
    # remove any negative weights caused by small negative transmissions
    integration_weights[integration_weights < 0.0] = 0.0

    spectrum = selden(
        electron_temperature[None, :, None],
        wavelength[:, None, None],
        scattering_angle[None, None, :],
        laser_wavelength=laser_wavelength,
    )
    return (spectrum * integration_weights[:, None, None]).sum(axis=0)


class SpectralResponse2D:
    def __init__(
        self, ln_te_axes: ndarray, scattering_angle_axes: ndarray, response: ndarray
    ):
        self.ln_te = ln_te_axes
        self.scattering_angle = scattering_angle_axes
        self.response = response

        # build the splines for all spatial / spectral channels
        self.n_positions, self.n_spectra, _, _ = self.response.shape
        self.splines = []
        for j in range(self.n_spectra):
            self.splines.append(
                [
                    RectBivariateSpline(
                        x=self.ln_te[i, :],
                        y=self.scattering_angle[i, :],
                        z=self.response[i, j, :, :],
                    )
                    for i in range(self.n_positions)
                ]
            )

    def get_response(self, ln_te: ndarray, scattering_angle: ndarray):
        y = zeros((self.n_positions, self.n_spectra, ln_te.shape[1]))
        for j in range(self.n_spectra):
            splines = self.splines[j]
            for i in range(self.n_positions):
                y[i, j, :] = splines[i].ev(ln_te[i, :], scattering_angle[i, :])
        return y

    def get_response_and_gradient(self, ln_te: ndarray, scattering_angle: ndarray):
        dS_dT = zeros((self.n_positions, self.n_spectra, ln_te.shape[1]))
        dS_dn = zeros((self.n_positions, self.n_spectra, ln_te.shape[1]))
        for j in range(self.n_spectra):
            splines = self.splines[j]
            for i in range(self.n_positions):
                dS_dn[i, j, :] = splines[i].ev(ln_te[i, :], scattering_angle[i, :])
                dS_dT[i, j, :] = splines[i].ev(
                    ln_te[i, :], scattering_angle[i, :], dx=1
                )
        return dS_dT, dS_dn

    def spectrum(self, Te: ndarray, ne: ndarray, weights: ndarray, scattering_angle: ndarray) -> ndarray:
        ln_te = log(Te)
        coeffs = ne * weights
        y = self.get_response(ln_te, scattering_angle)
        y *= coeffs[:, None, :]
        return y.sum(axis=2)

    def spectrum_jacobian(self, Te: ndarray, ne: ndarray, weights: ndarray, scattering_angle: ndarray):
        ln_te = log(Te)
        coeffs = ne * weights
        dS_dT, dS_dn = self.get_response_and_gradient(
            ln_te, scattering_angle
        )
        dS_dT *= (coeffs / Te)[:, None, :]
        dS_dn *= weights[:, None, :]
        return dS_dT, dS_dn

    @classmethod
    def calculate_response(
        cls,
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


class SpectralResponse:
    def __init__(
        self,
        ln_te: ndarray,
        scattering_angle: ndarray,
        response: ndarray,
        scattering_angle_gradient: ndarray,
        ln_te_gradient: ndarray,
        ln_te_and_angle_gradient: ndarray,
        jit_compile=False,
    ):
        self.ln_te_knots = ln_te.astype(float32)
        self.knot_spacing = ln_te[1] - ln_te[0]
        self.scattering_angle = scattering_angle.astype(float32)
        self.angle_grad = scattering_angle_gradient
        self.response = response

        # build the splines for all spatial / spectral channels
        self.n_positions, self.n_spectra, self.n_knots = self.response.shape

        self.y = stack(
            arrays=[
                response,
                scattering_angle_gradient,
                ln_te_gradient,
                ln_te_and_angle_gradient,
            ],
            axis=3,
            dtype=float32,
        )
        coeff_shape = (self.n_positions, self.n_spectra, self.n_knots - 1, 4)
        self.a = zeros(coeff_shape, dtype=float32)
        self.b = zeros(coeff_shape, dtype=float32)

        for i in range(self.n_positions):
            for j in range(self.n_spectra):
                for k in range(4):
                    a, b = cubic_spline_coefficients(
                        self.ln_te_knots, self.y[i, j, :, k]
                    )
                    self.a[i, j, :, k] = a
                    self.b[i, j, :, k] = b

        self.a_no_grads = self.a[:, :, :, :2].copy()
        self.b_no_grads = self.b[:, :, :, :2].copy()
        self.y_no_grads = self.y[:, :, :, :2].copy()


        if jit_compile == "numba":
            import pedinf.spectrum.numba as spec
        elif jit_compile == "jax":
            import pedinf.spectrum.jax as spec
        else:
            import pedinf.spectrum.numpy as spec


        self.__spectrum = spec.spectrum
        self.__spectrum_jacobian = spec.spectrum_jacobian
        self.__response = spec.response
        self.__response_and_grad = spec.response_and_grad

    def spectrum(
        self, Te: ndarray, ne: ndarray, weights: ndarray, scattering_angle: ndarray
    ):
        return self.__spectrum(
            Te,
            ne,
            scattering_angle - self.scattering_angle[:, None],
            weights,
            self.y_no_grads,
            self.a_no_grads,
            self.b_no_grads,
            self.ln_te_knots,
            self.knot_spacing,
        )

    def spectrum_jacobian(
        self, Te: ndarray, ne: ndarray, weights: ndarray, scattering_angle: ndarray
    ):
        return self.__spectrum_jacobian(
            Te,
            ne,
            scattering_angle - self.scattering_angle[:, None],
            weights,
            self.y,
            self.a,
            self.b,
            self.ln_te_knots,
            self.knot_spacing,
        )

    def response_and_grad(
        self, ln_te: ndarray, scattering_angle: ndarray
    ):
        return self.__response_and_grad(
            ln_te,
            scattering_angle - self.scattering_angle[:, None],
            self.y,
            self.a,
            self.b,
            self.ln_te_knots,
            self.knot_spacing,
        )

    @classmethod
    def calculate_response(
        cls,
        wavelengths: ndarray,
        transmissions: ndarray,
        scattering_angles: ndarray,
        spatial_channels: ndarray,
        ln_te_range=(-3, 10),
        n_temps=128,
        jit_compile=False,
    ):
        response_data = SpectralResponse.calculate_response_data(
            wavelengths=wavelengths,
            transmissions=transmissions,
            scattering_angles=scattering_angles,
            spatial_channels=spatial_channels,
            ln_te_range=ln_te_range,
            n_temps=n_temps,
        )
        return cls(**response_data, jit_compile=jit_compile)

    @staticmethod
    def calculate_response_data(
        wavelengths: ndarray,
        transmissions: ndarray,
        scattering_angles: ndarray,
        spatial_channels: ndarray,
        ln_te_range=(-3, 10),
        n_temps=128,
    ) -> dict:
        n_spectral_chans = 4
        ln_te = linspace(*ln_te_range, n_temps)
        te_axis = exp(ln_te)

        response = zeros([spatial_channels.size, n_spectral_chans, n_temps])
        angle_grad = zeros([spatial_channels.size, n_spectral_chans, n_temps])
        temp_grad = zeros([spatial_channels.size, n_spectral_chans, n_temps])
        double_grad = zeros([spatial_channels.size, n_spectral_chans, n_temps])
        dA = 1e-6
        dT = 1e-4

        for i, chan in enumerate(spatial_channels):
            angle_axis = scattering_angles[chan] + array([0.0, -dA, dA])
            for j in range(n_spectral_chans):
                f0, f1, f2 = calculate_filter_response(
                    electron_temperature=te_axis,
                    scattering_angle=angle_axis,
                    wavelength=wavelengths[chan, j, :],
                    transmission=transmissions[chan, j, :],
                ).T

                response[i, j, :] = f0
                angle_grad[i, j, :] = 0.5 * (f2 - f1) / dA

                dte_1, f1, f2 = calculate_filter_response(
                    electron_temperature=exp(ln_te - dT),
                    scattering_angle=angle_axis,
                    wavelength=wavelengths[chan, j, :],
                    transmission=transmissions[chan, j, :],
                ).T
                dphi_1 = 0.5 * (f2 - f1) / dA

                dte_2, f1, f2 = calculate_filter_response(
                    electron_temperature=exp(ln_te + dT),
                    scattering_angle=angle_axis,
                    wavelength=wavelengths[chan, j, :],
                    transmission=transmissions[chan, j, :],
                ).T
                dphi_2 = 0.5 * (f2 - f1) / dA

                temp_grad[i, j, :] = 0.5 * (dte_2 - dte_1) / dT
                double_grad[i, j, :] = 0.5 * (dphi_2 - dphi_1) / dT

        return dict(
            ln_te=ln_te,
            scattering_angle=scattering_angles[spatial_channels],
            response=response,
            scattering_angle_gradient=angle_grad,
            ln_te_gradient=temp_grad,
            ln_te_and_angle_gradient=double_grad
        )
