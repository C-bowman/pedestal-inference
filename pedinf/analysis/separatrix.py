from numpy import mean, std, isfinite
from numpy import linspace, zeros, ndarray
from numpy.random import normal
from pedinf.models import ProfileModel
from pedinf.analysis.utils import locate_radius


def separatrix_given_temperature(
    ne_profile_samples: ndarray,
    te_profile_samples: ndarray,
    model: ProfileModel,
    te_sep: float = None,
    te_sep_error: float = None,
    te_sep_samples: ndarray = None,
) -> dict[str, float]:
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
        An instance of one of the model classes from ``pedinf.models``.

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
        te_sep_samples = te_sep_samples[
            abs(te_sep_samples - te_sep) < 2.5 * te_sep_error
        ]
    assert te_sep_samples.size >= n_prof

    for i in range(n_prof):
        R = locate_radius(te_sep_samples[i], te_profile_samples[i, :], model=model)
        ne_sep_samples[i] = model.prediction(R, ne_profile_samples[i, :])
        pe_sep_samples[i] = ne_sep_samples[i] * te_sep_samples[i]
        R_sep_samples[i] = R

    return {
        "ne_mean": mean(ne_sep_samples),
        "ne_std": std(ne_sep_samples),
        "pe_mean": mean(pe_sep_samples),
        "pe_std": std(pe_sep_samples),
        "R_mean": mean(R_sep_samples),
        "R_std": std(R_sep_samples),
    }


def linear_find_zero(x1: float, x2: float, y1: float, y2: float) -> float:
    return x1 - y1 * (x2 - x1) / (y2 - y1)


def separatrix_given_scaling(
    ne_profile_samples: ndarray,
    te_profile_samples: ndarray,
    model: ProfileModel,
    separatrix_scaling: callable,
    radius_limits=(1.2, 1.6),
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
        An instance of one of the model classes from ``pedinf.models``.

    :param callable separatrix_scaling: \
        A callable which maps a given separatrix density to a corresponding
        separatrix temperature. It must take an array of electron density
        values as its only argument.

    :param radius_limits: \
        A tuple specifying the range of radius values used to search for the
        separatrix position in the form ``(lower_limit, upper_limit)``.

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
        i, j = (m - 1, m) if dt[m] * dt[m - 1] < 0.0 else (m, m + 1)
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
        "ne_mean": mean(ne_sep_samples),
        "ne_std": std(ne_sep_samples),
        "te_mean": mean(te_sep_samples),
        "te_std": std(te_sep_samples),
        "pe_mean": mean(pe_sep_samples),
        "pe_std": std(pe_sep_samples),
        "R_mean": mean(R_sep_samples),
        "R_std": std(R_sep_samples),
    }
