from numpy import zeros, ndarray
from pedinf.models import ProfileModel
from pedinf.analysis.utils import locate_radius, PlasmaProfile


def profiles_by_temperature(
        ne_profile_samples: ndarray,
        te_profile_samples: ndarray,
        model: ProfileModel,
        temperatures: ndarray
):
    """
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

    """
    n_samp, n_params = te_profile_samples.shape
    radius_samples = zeros([temperatures.size, n_samp])
    density_samples = zeros([temperatures.size, n_samp])
    for i in range(n_samp):
        radius_samples[:, i] = locate_radius(
            profile_values=temperatures,
            model=model,
            parameters=te_profile_samples[i, :],
            search_points=50,
            show_warnings=False
        )

        density_samples[:, i] = model.prediction(radius_samples[:, i], ne_profile_samples[i, :])

    pressure_samples = density_samples * temperatures[:, None]

    pressure = PlasmaProfile(
        axis=temperatures,
        profile_samples=pressure_samples,
        axis_label="electron temperature",
        axis_units="eV",
        profile_label="electron pressure",
        profile_units="eV / m^3"
    )

    density = PlasmaProfile(
        axis=temperatures,
        profile_samples=density_samples,
        axis_label="electron temperature",
        axis_units="eV",
        profile_label="electron density",
        profile_units="m^-3"
    )

    radius = PlasmaProfile(
        axis=temperatures,
        profile_samples=radius_samples,
        axis_label="electron temperature",
        axis_units="eV",
        profile_label="major radius",
        profile_units="m"
    )

    return radius, density, pressure
