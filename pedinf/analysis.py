from numpy import linspace, mean, ndarray, std, sqrt, isfinite, zeros
from typing import Type
from functools import partial
from numpy.random import normal
from scipy.optimize import minimize
from warnings import warn
from pedinf.models import ProfileModel
from inference.pdf import sample_hdi
from inference.likelihoods import GaussianLikelihood


def locate_radius(
        profile_value: float,
        theta: ndarray,
        model: Type[ProfileModel],
        search_limits=(1.2, 1.5),
        tolerance=1e-4
):
    if profile_value <= 0. or not isfinite(profile_value):
        raise ValueError(
            f"""\n
            [ locate_radius error ]
            >> 'profile_value' argument must be finite and greater than zero.
            >> The given value was {profile_value}
            """
        )

    R_search = linspace(*search_limits, 21)
    profile = model.prediction(R_search, theta)
    cut = profile.argmax()
    R_search = R_search[cut:]
    profile = profile[cut:]

    index = abs(profile_value - profile).argmin()
    R = R_search[index]
    initial_R = R.copy()
    initial_val = profile[index]
    initial_err = abs((profile_value - initial_val) / profile_value)

    if not isfinite(initial_val):
        raise ValueError(
            f"""\n
            [ locate_radius error ]
            >> The initial estimate of the radius value produced by the grid
            >> search yields a non-finite profile value of {initial_val}
            """
        )

    for i in range(8):
        dy = (profile_value - model.prediction(R, theta))
        R += dy / model.gradient(R, theta)
        if abs(dy / profile_value) < tolerance:
            break
    else:
        R = initial_R
        warn(
            f"""\n
            [ locate_radius warning ]
            >> Newton iteration failed to converge. Instead returning grid-search
            >> estimate of {initial_R} with a fractional error of {initial_err}.
            """
        )
    return R


def separatrix_given_temperature(
        ne_profile_samples: ndarray,
        te_profile_samples: ndarray,
        model: Type[ProfileModel],
        te_sep: float = None,
        te_sep_error: float = None,
        te_sep_samples: ndarray = None
):
    """
    Given an estimated separatrix electron temperature (and optionally an associated
    uncertainty) this function estimates the separatrix major radius, electron density
    and electron pressure.

    :param ne_profile_samples: \
        A set of sampled parameters of a profile model from ``pedinf.models``
        representing possible electron density edge profiles. The samples should
        be given as a ``numpy.ndarray`` of shape ``(n, m)`` where ``n`` is the number
        of samples and ``m`` is the number of model parameters.

    :param te_profile_samples: \
        A set of sampled parameters of a profile model from ``pedinf.models``
        representing possible electron temperature edge profiles. The samples should
        be given as a ``numpy.ndarray`` of shape ``(n, m)`` where ``n`` is the number
        of samples and ``m`` is the number of model parameters.

    :param model: \
        A profile model from the ``pedinf.models`` module.

    :param te_sep: \
        The electron temperature value for which the corresponding major radius,
        electron density and electron pressure will be estimated.

    :param te_sep_error: \
        The Gaussian uncertainty on the given value of ``te_sep``.

    :param te_sep_samples: \
        Samples of the separatrix temperature value. This allows non-gaussian uncertainties
        on the separatrix temperature to included in the analysis. Specifying ``te_sep_samples``
        overrides any values given to the ``te_sep`` or ``te_sep_error`` arguments, which will
        be ignored. The number of separatrix temperature samples given must be equal to the
        number of profile samples.

    :return: \
        A dictionary containing the mean and standard-deviation for the separatrix major
        radius, separatrix electron density, and separatrix electron pressure. The keys
        are ``R_mean, ne_mean, pe_mean`` for the means, and
        ``R_std, ne_std, pe_std`` for the standard-deviations.
    """
    if ne_profile_samples.shape != te_profile_samples.shape:
        raise ValueError(
            """\n
            [ separatrix_given_temperature error ]
            >> The 'ne_profile_samples' and 'te_profile_samples' arrays must
            >> have the same shape.
            """
        )

    if not isfinite(ne_profile_samples).all() or not isfinite(te_profile_samples).all():
        raise ValueError(
            """\n
            [ separatrix_given_temperature error ]
            >> The 'ne_profile_samples' and/or 'te_profile_samples' arrays contain non-finite values.
            """
        )

    if te_sep is None and te_sep_samples is None:
        raise ValueError(
            """\n
            [ separatrix_given_temperature error ]
            >> At least one of the 'te_sep' and 'te_sep_samples' arguments
            >> must be specified.
            """
        )

    if model.n_parameters != ne_profile_samples.shape[1]:
        raise ValueError(
            f"""\n
            [ separatrix_given_temperature error ]
            >> The given samples have {ne_profile_samples.shape[1]} parameters, but the
            >> given '{model.name}' profile model has {model.n_parameters} parameters.
            """
        )

    te_sep_error = 0.1 if te_sep_error is None else te_sep_error
    n_prof = ne_profile_samples.shape[0]
    ne_sep_samples = zeros(n_prof)
    pe_sep_samples = zeros(n_prof)
    R_sep_samples = zeros(n_prof)

    if te_sep_samples is None:
        te_sep_samples = normal(loc=te_sep, scale=te_sep_error, size=int(n_prof * 1.2))
        te_sep_samples = te_sep_samples[abs(te_sep_samples - te_sep) < 2.5 * te_sep_error]
    assert te_sep_samples.size >= n_prof

    for i in range(n_prof):
        R = locate_radius(te_sep_samples[i], te_profile_samples[i, :], model=model)
        ne_sep_samples[i] = model.prediction(R, ne_profile_samples[i, :])
        pe_sep_samples[i] = ne_sep_samples[i] * te_sep_samples[i]
        R_sep_samples[i] = R

    return {
        'ne_mean': mean(ne_sep_samples),
        'ne_std': std(ne_sep_samples),
        'pe_mean': mean(pe_sep_samples),
        'pe_std': std(pe_sep_samples),
        'R_mean': mean(R_sep_samples),
        'R_std': std(R_sep_samples)
    }


def linear_find_zero(x1, x2, y1, y2):
    return x1 - y1 * (x2 - x1) / (y2 - y1)


def separatrix_given_scaling(
        ne_profile_samples: ndarray,
        te_profile_samples: ndarray,
        model: Type[ProfileModel],
        separatrix_scaling: callable,
        radius_limits=(1.2, 1.6)
):
    """
    Given a scaling function which specifies the separatrix temperature as
    a function of the separatrix density, estimate the mean and standard-deviation
    of the separatrix radius and separatrix electron density, temperature and pressure.

    :param ne_profile_samples: \
        A set of sampled parameters of a profile model from ``pedinf.models``
        representing possible electron density edge profiles. The samples should
        be given as a ``numpy.ndarray`` of shape ``(n, m)`` where ``n`` is the number
        of samples and ``m`` is the number of model parameters.

    :param te_profile_samples: \
        A set of sampled parameters of a profile model from ``pedinf.models``
        representing possible electron temperature edge profiles. The samples should
        be given as a ``numpy.ndarray`` of shape ``(n, m)`` where ``n`` is the number
        of samples and ``m`` is the number of model parameters.

    :param model: \
        A profile model from the ``pedinf.models`` module.

    :param callable separatrix_scaling: \
        A callable which maps a given separatrix density to a corresponding
        separatrix temperature. It must take an array of electron density
        values as its only argument.

    :return: \
        A dictionary containing the mean and standard-deviation for the
        separatrix major radius, and separatrix electron temperature,
        density and pressure. The keys are ``R_mean, te_mean, ne_mean, pe_mean``
        for the means, and ``R_std, te_std, ne_std, pe_std`` for the
        standard-deviations.
    """

    # should put some input parsing here to verify the shapes of the given sample arrays
    ne_sep_samples = []
    te_sep_samples = []
    pe_sep_samples = []
    R_sep_samples = []

    # loop over all the samples for this TS profile
    radius_axis = linspace(*radius_limits, 128)
    for smp in range(ne_profile_samples.shape[0]):
        # impose the SOLPS scaling to find the separatrix position
        te_prof = model.prediction(radius_axis, te_profile_samples[smp, :])
        ne_prof = model.prediction(radius_axis, ne_profile_samples[smp, :])
        te_sep_prediction = separatrix_scaling(ne_prof)
        dt = te_prof - te_sep_prediction
        m = abs(dt).argmin()
        i, j = (m - 1, m) if dt[m] * dt[m - 1] < 0. else (m, m + 1)
        R_sep = linear_find_zero(radius_axis[i], radius_axis[j], dt[i], dt[j])

        # use separatrix position to get the temperature / density
        te_sep = model.prediction(R_sep, te_profile_samples[smp, :])
        ne_sep = model.prediction(R_sep, ne_profile_samples[smp, :])

        # store the results for this sample
        ne_sep_samples.append(ne_sep)
        te_sep_samples.append(te_sep)
        pe_sep_samples.append(ne_sep * te_sep)
        R_sep_samples.append(R_sep)

    return {
        'ne_mean': mean(ne_sep_samples),
        'ne_std': std(ne_sep_samples),
        'te_mean': mean(te_sep_samples),
        'te_std': std(te_sep_samples),
        'pe_mean': mean(pe_sep_samples),
        'pe_std': std(pe_sep_samples),
        'R_mean': mean(R_sep_samples),
        'R_std': std(R_sep_samples)
    }


def pressure_profile_and_gradient(
        radius: ndarray,
        ne_profile_samples: ndarray,
        te_profile_samples: ndarray,
        model: Type[ProfileModel]
):
    """
    Calculates the electron pressure and pressure gradient profiles at specified major
    radius positions, given samples of the edge electron temperature and density profiles.

    It is assumed that the electron temperatures and densities have units of :math:`eV`
    and :math:`m^{-3}` respectively, and the returned electron pressure will have units
    of :math:`N / m^2`.

    :param radius: \
        Major radius values at which to evaluate the pressure profiles as
        a ``numpy.ndarray``.

    :param ne_profile_samples: \
        A set of sampled parameters of a profile model from ``pedinf.models``
        representing possible electron density edge profiles. The samples should
        be given as a ``numpy.ndarray`` of shape ``(n, m)`` where ``n`` is the number
        of samples and ``m`` is the number of model parameters.

    :param te_profile_samples: \
        A set of sampled parameters of a profile model from ``pedinf.models``
        representing possible electron temperature edge profiles. The samples should
        be given as a ``numpy.ndarray`` of shape ``(n, m)`` where ``n`` is the number
        of samples and ``m`` is the number of model parameters.

    :param model: \
        A profile model from the ``pedinf.models`` module.

    :return: \
        A dictionary of results with the following keys:

            - ``"radius"`` : The given radius axis on which the pressure profile was evaluated.
            - ``"pe_profiles"`` : The sampled pressure profiles as a 2D ``numpy.ndarray``.
            - ``"pe_mean"`` : The mean of the posterior predictive distribution for the pressure.
            - ``"pe_hdi_65"`` : The 65% highest-density interval for the pressure profile.
            - ``"pe_hdi_95"`` : The 95% highest-density interval for the pressure profile.
            - ``"pe_gradient_profiles"`` : The sampled pressure gradient profiles as a 2D ``numpy.ndarray``.
            - ``"pe_gradient_mean"`` : The mean of the posterior predictive distribution for the pressure gradient.
            - ``"pe_gradient_hdi_65"`` : The 65% highest-density interval for the pressure gradient profile.
            - ``"pe_gradient_hdi_95"`` : The 95% highest-density interval for the pressure gradient profile.
    """
    n_samples = ne_profile_samples.shape[0]
    pe_profs = zeros([n_samples, radius.size])
    pe_grads = zeros([n_samples, radius.size])
    for smp in range(n_samples):
        te_prof = model.prediction(radius, te_profile_samples[smp, :])
        te_grad = model.gradient(radius, te_profile_samples[smp, :])
        ne_prof = model.prediction(radius, ne_profile_samples[smp, :])
        ne_grad = model.gradient(radius, ne_profile_samples[smp, :])

        pe_profs[smp, :] = te_prof * ne_prof
        pe_grads[smp, :] = te_prof * ne_grad + ne_prof * te_grad

    # convert pressure from eV / m^3 to J / m^3 by multiplying by electron charge
    pe_profs *= 1.60217663e-19
    pe_grads *= 1.60217663e-19

    return {
        "radius": radius,
        "pe_profiles": pe_profs,
        "pe_mean": pe_profs.mean(axis=0),
        "pe_hdi_65": sample_hdi(pe_profs, fraction=0.65),
        "pe_hdi_95": sample_hdi(pe_profs, fraction=0.95),
        "pe_gradient_profiles": pe_grads,
        "pe_gradient_mean": pe_grads.mean(axis=0),
        "pe_gradient_hdi_65": sample_hdi(pe_grads, fraction=0.65),
        "pe_gradient_hdi_95": sample_hdi(pe_grads, fraction=0.95),
    }


def pressure_parameters(
        ne_parameters: ndarray,
        te_parameters: ndarray,
        model: Type[ProfileModel],
        return_diagnostics=False
):
    """
    Approximates the electron pressure profile parameters by re-fitting the profile
    model to the predicted pressure profile obtained from the product of the predicted
    temperature and density profiles.

    :param ne_parameters: \
        The density parameter vector for the given profile model.

    :param te_parameters: \
        The temperature parameter vector for the given profile model.

    :param model: \
        Pass either the ``mtanh` or ``lpm`` models from ``pedinf.models``.

    :param return_diagnostics: \
        If set as ``True``, a dictionary containing diagnostic information about
        the fit to the pressure will be returned alongside the fit parameters.

    :return: \
        The fitted pressure profile parameters.
    """

    R0_index = model.parameters["pedestal_location"]
    w_index = model.parameters["pedestal_width"]
    ne_R0, te_R0 = ne_parameters[R0_index], te_parameters[R0_index]
    ne_w, te_w = ne_parameters[w_index], te_parameters[w_index]

    # use te / ne parameters to determine the range over which to fit the pressure
    n_points = 128
    R_min = min(ne_R0, te_R0) - 2*(ne_w + te_w)
    R_max = max(ne_R0, te_R0) + 2*(ne_w + te_w)
    R = linspace(R_min, R_max, n_points)

    ec = 1.60217663e-19  # electron charge used to convert from eV / m^3 to J / m^3
    pe_prediction = (model.prediction(R, ne_parameters) * model.prediction(R, te_parameters)) * ec
    forward_model = partial(model.prediction, R)
    forward_model_jacobian = partial(model.jacobian, R)

    # build an initial guess for the pressure parameters
    initial_guess = 0.5 * (ne_parameters + te_parameters)
    i = model.parameters["pedestal_height"]
    j = model.parameters["pedestal_top_gradient"]
    k = model.parameters["background_level"]
    initial_guess[i] = ne_parameters[i] * te_parameters[i] * ec
    initial_guess[j] = (ne_parameters[i] * te_parameters[j] + ne_parameters[j] * te_parameters[i]) * ec
    initial_guess[k] = ne_parameters[k] * te_parameters[k] * ec
    # set location guess to be position of maximum gradient in the pressure prediction
    initial_guess[R0_index] = R[1:-1][(pe_prediction[2:] - pe_prediction[:-2]).argmin()]

    # set up a gaussian likelihood function we can maximise to estimate the pressure
    L = GaussianLikelihood(
        y_data=pe_prediction,
        sigma=zeros(n_points) + initial_guess[i]*0.01,
        forward_model=forward_model,
        forward_model_jacobian=forward_model_jacobian
    )

    bounds = {
        "pedestal_location": (0., None),
        "pedestal_height": (0., None),
        "pedestal_width": (1e-3, None),
        "pedestal_top_gradient": (None, None),
        "background_level": (0., None),
        "logistic_shape_parameter": (0.05, None),
    }

    result = minimize(
        fun=L.cost,
        x0=initial_guess,
        method="Nelder-Mead",
        bounds=[bounds[p] for p in model.parameters],
        options={"maxiter": 3000}
    )

    if return_diagnostics:
        fit = forward_model(result.x)
        residual = pe_prediction - fit
        diagnostics = {
            "max_abs_err": abs(residual).max(),
            "rmse": sqrt((residual**2).mean()),
            "target": pe_prediction,
            "fit": fit,
            "radius": R
        }
        return result.x, diagnostics
    else:
        return result.x