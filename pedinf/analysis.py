from numpy import linspace, mean, std, isfinite, zeros
from numpy.random import normal
from warnings import warn
from pedinf.model import mtanh, mtanh_gradient
from inference.pdf import sample_hdi


def locate_radius(profile_value, theta, tolerance=1e-4):
    if profile_value <= theta[1]*theta[4]:
        raise ValueError(
            f"""
            [ locate_radius error ]
            >> 'profile_value' of {profile_value} is below the profile
            >> minimum value of {theta[1]*theta[4]}.
            """
        )
    if profile_value <= 0. or not isfinite(profile_value):
        raise ValueError(
            f"""
            [ locate_radius error ]
            >> 'profile_value' argument must be finite and greater than zero.
            >> The given value was {profile_value}
            """
        )

    if mtanh(theta[0], theta) > profile_value:
        R_search = linspace(theta[0], theta[0] + 3 * theta[2], 21)
    else:
        R_search = linspace(theta[0] - 3 * theta[2], theta[0], 21)

    R = R_search[abs(profile_value - mtanh(R_search, theta)).argmin()]
    initial_R = R.copy()
    initial_val = mtanh(initial_R, theta)
    initial_err = abs((profile_value - initial_val) / profile_value)

    if not isfinite(initial_val):
        raise ValueError(
            f"""
            [ locate_radius error ]
            >> The initial estimate of the radius value produced by the grid
            >> search yields a non-finite profile value of {initial_val}
            """
        )

    for i in range(8):
        dy = (profile_value - mtanh(R, theta))
        R += dy / mtanh_gradient(R, theta)
        if abs(dy / profile_value) < tolerance:
            break
    else:
        R = initial_R
        warn(
            f"""
            [ locate_radius warning ]
            >> Newton iteration failed to converge. Instead returning grid-search
            >> estimate of {initial_R} with a fractional error of {initial_err}.
            """
        )
    return R


def separatrix_from_temperature(ne_samples, te_samples, te_value, te_error=None):
    """
    Given an estimated separatrix electron temperature (and optionally an associated
    uncertainty) this function estimates the separatrix major radius, electron density
    and electron pressure.

    :param ne_samples: \
        A set of sampled parameters of the ``mtanh`` function representing
        possible electron density edge profiles. The samples should be given
        as a ``numpy.ndarray`` of shape ``(n, 6)`` where ``n`` is the number
        of samples.

    :param te_samples: \
        A set of sampled parameters of the ``mtanh`` function representing
        possible electron temperature edge profiles. The samples should be given
        as a ``numpy.ndarray`` of shape ``(n, 6)`` where ``n`` is the number
        of samples.

    :param te_value: \
        The electron temperature value for which the corresponding major radius,
        electron density and electron pressure will be estimated.

    :param te_error: \
        The uncertainty on the given value of ``te_value``.

    :return: \
        A dictionary containing the mean and standard-deviation for the separatrix major
        radius, separatrix electron density, and separatrix electron pressure. The keys
        are ``R_mean, ne_mean, pe_mean`` for the means, and
        ``R_std, ne_std, pe_std`` for the standard-deviations.
    """
    if not isfinite(ne_samples).all() or not isfinite(te_samples).all():
        raise ValueError(
            """
            [ density_given_temperature error ]
            >> The 'ne_samples' and/or 'te_samples' arrays contain non-finite values.
            """
        )
    te_error = 0. if te_error is None else te_error
    n_prof = ne_samples.shape[0]
    ne_sep_samples = zeros(n_prof)
    pe_sep_samples = zeros(n_prof)
    R_sep_samples = zeros(n_prof)
    for i in range(n_prof):
        te = te_value + te_error*normal()
        R = locate_radius(te, te_samples[i, :])
        ne_sep_samples[i] = mtanh(R, ne_samples[i, :])
        pe_sep_samples[i] = ne_sep_samples[i] * te
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


def separatrix_from_scaling(ne_samples, te_samples, separatrix_scaling, radius_limits=(1.2, 1.6)):
    """
    Given a scaling function which specifies the separatrix temperature as
    a function of the separatrix density, estimates the mean and standard-deviation
    of the separatrix radius and separatrix electron density, temperature and pressure.

    :param ne_samples: \
        A set of sampled parameters of the ``mtanh`` function representing
        possible electron density edge profiles. The samples should be given
        as a ``numpy.ndarray`` of shape ``(n, 6)`` where ``n`` is the number
        of samples.

    :param te_samples: \
        A set of sampled parameters of the ``mtanh`` function representing
        possible electron temperature edge profiles. The samples should be given
        as a ``numpy.ndarray`` of shape ``(n, 6)`` where ``n`` is the number
        of samples.

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

    # should put some input parsing here to verify the shapes
    # of the given sample arrays
    ne_sep_samples = []
    te_sep_samples = []
    pe_sep_samples = []
    R_sep_samples = []

    # loop over all the samples for this TS profile
    radius_axis = linspace(radius_limits[0], radius_limits[1], 128)
    for smp in range(ne_samples.shape[0]):
        # impose the SOLPS scaling to find the separatrix position
        te_prof = mtanh(radius_axis, te_samples[smp, :])
        te_sep_prediction = separatrix_scaling(mtanh(radius_axis, ne_samples[smp, :]))
        dt = te_prof - te_sep_prediction
        m = abs(dt).argmin()
        i, j = (m - 1, m) if dt[m] * dt[m - 1] < 0. else (m, m + 1)
        R_sep = linear_find_zero(radius_axis[i], radius_axis[j], dt[i], dt[j])

        # use separatrix position to get the temperature / density
        te_sep = mtanh(R_sep, te_samples[smp, :])
        ne_sep = mtanh(R_sep, ne_samples[smp, :])

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


def pressure_profile_and_gradient(radius, ne_samples, te_samples):
    """
    ...

    :param radius: \
        ...

    :param ne_samples: \
        A set of sampled parameters of the ``mtanh`` function representing
        possible electron density edge profiles. The samples should be given
        as a ``numpy.ndarray`` of shape ``(n, 6)`` where ``n`` is the number
        of samples.

    :param te_samples: \
        A set of sampled parameters of the ``mtanh`` function representing
        possible electron temperature edge profiles. The samples should be given
        as a ``numpy.ndarray`` of shape ``(n, 6)`` where ``n`` is the number
        of samples.

    :return: \
        ...
    """
    n_samples = ne_samples.shape[0]
    pe_profs = zeros([n_samples, radius.size])
    pe_grads = zeros([n_samples, radius.size])
    for smp in range(n_samples):
        te_prof = mtanh(radius, te_samples[smp, :])
        te_grad = mtanh_gradient(radius, te_samples[smp, :])
        ne_prof = mtanh(radius, ne_samples[smp, :])
        ne_grad = mtanh_gradient(radius, ne_samples[smp, :])

        pe_profs[smp, :] = te_prof * ne_prof
        pe_grads[smp, :] = te_prof * ne_grad + ne_prof * te_grad

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