from numpy import linspace, mean, std, zeros
from numpy.random import normal
from pedinf.model import mtanh, mtanh_gradient


def locate_radius(profile_value, theta, tolerance=1e-4):
    if profile_value <= theta[1]*theta[4]:
        raise ValueError("target value below minimum")

    R_search = linspace(theta[0] - 3*theta[2], theta[0] + 3*theta[2], 21)
    R = R_search[abs(profile_value - mtanh(R_search, theta)).argmin()]

    for i in range(20):
        dy = (profile_value - mtanh(R, theta))
        R += dy / mtanh_gradient(R, theta)
        if abs(dy / profile_value) < tolerance:
            break
    else:
        raise ValueError(
            f"""
            >> Failed to find radius at which given profile values occurs.
            >> Lowest error was {abs(dy / profile_value)}, but tolerance is {tolerance}.
            """
        )
    return R


def density_given_temperature(ne_samples, te_samples, te_value, te_error=None):
    te_error = 0. if te_error is None else te_error
    n_prof = ne_samples.shape[0]
    density_samples = zeros(n_prof)
    for i in range(n_prof):
        te = te_value + te_error*normal()
        R = locate_radius(te, te_samples[i, :])
        density_samples[i] = mtanh(R, ne_samples[i, :])
    return density_samples


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
        density and pressure. The keys are ``'R_mean', te_mean, ne_mean, pe_mean``
        for the means, and ``'R_std', te_std, ne_std, pe_std`` for the
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


def pressure_profile_and_gradient(ne_samples, te_samples):
    pass
