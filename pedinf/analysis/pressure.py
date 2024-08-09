from numpy import sqrt
from numpy import linspace, zeros, ndarray
from functools import partial
from scipy.optimize import minimize
from pedinf.models import ProfileModel
from inference.likelihoods import GaussianLikelihood
from pedinf.analysis.profile import PlasmaProfile


def pressure_profile_and_gradient(
    radius: ndarray,
    ne_profile_samples: ndarray,
    te_profile_samples: ndarray,
    model: ProfileModel,
) -> tuple[PlasmaProfile, PlasmaProfile]:
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
        An instance of one of the model classes from ``pedinf.models``.

    :return: \
        A tuple of two ``PlasmaProfile`` objects, the first corresponding to the
        electron pressure profile, and the second to the electron pressure gradient
        profile.
    """
    # make a copy of the model instance so we can avoid changing the original
    model = model.from_configuration(model.get_model_configuration())
    model.update_radius(radius)
    n_samples = ne_profile_samples.shape[0]
    pe_profs = zeros([n_samples, radius.size])
    pe_grads = zeros([n_samples, radius.size])
    for smp in range(n_samples):
        te_prof = model.forward_prediction(te_profile_samples[smp, :])
        te_grad = model.forward_gradient(te_profile_samples[smp, :])
        ne_prof = model.forward_prediction(ne_profile_samples[smp, :])
        ne_grad = model.forward_gradient(ne_profile_samples[smp, :])

        pe_profs[smp, :] = te_prof * ne_prof
        pe_grads[smp, :] = te_prof * ne_grad + ne_prof * te_grad

    # convert pressure from eV / m^3 to J / m^3 by multiplying by electron charge
    pe_profs *= 1.60217663e-19
    pe_grads *= 1.60217663e-19

    pe_profile = PlasmaProfile(
        axis=radius,
        profile_samples=pe_profs.T,
        axis_label="major radius",
        axis_units="m",
        profile_label="electron pressure",
        profile_units="N / m^2"
    )

    pe_gradient = PlasmaProfile(
        axis=radius,
        profile_samples=pe_grads.T,
        axis_label="major radius",
        axis_units="m",
        profile_label="electron pressure gradient",
        profile_units="N / m^3"
    )

    return pe_profile, pe_gradient


def pressure_parameters(
    ne_parameters: ndarray,
    te_parameters: ndarray,
    model: ProfileModel,
    return_diagnostics=False,
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
        An instance of either the ``mtanh` or ``lpm`` model classes from ``pedinf.models``.

    :param return_diagnostics: \
        If set as ``True``, a dictionary containing diagnostic information about
        the fit to the pressure will be returned alongside the fit parameters.

    :return: \
        The fitted pressure profile parameters.
    """
    if model.name not in ["mtanh", "lpm"]:
        raise ValueError(
            """\n
            [ pressure_parameters error ]
            >> 'model' argument must be an instance of either the
            >> 'mtanh' or 'lpm' model classes.
            """
        )

    R0_index = model.parameters["pedestal_location"]
    w_index = model.parameters["pedestal_width"]
    ne_R0, te_R0 = ne_parameters[R0_index], te_parameters[R0_index]
    ne_w, te_w = ne_parameters[w_index], te_parameters[w_index]

    # use te / ne parameters to determine the range over which to fit the pressure
    n_points = 128
    R_min = min(ne_R0, te_R0) - 2 * (ne_w + te_w)
    R_max = max(ne_R0, te_R0) + 2 * (ne_w + te_w)
    R = linspace(R_min, R_max, n_points)

    ec = 1.60217663e-19  # electron charge used to convert from eV / m^3 to J / m^3
    pe_prediction = (
        model.prediction(R, ne_parameters) * model.prediction(R, te_parameters)
    ) * ec
    forward_model = partial(model.prediction, R)
    forward_model_jacobian = partial(model.jacobian, R)

    # build an initial guess for the pressure parameters
    initial_guess = 0.5 * (ne_parameters + te_parameters)
    i = model.parameters["pedestal_height"]
    j = model.parameters["pedestal_top_gradient"]
    k = model.parameters["background_level"]
    initial_guess[i] = ne_parameters[i] * te_parameters[i] * ec
    initial_guess[j] = (
        ne_parameters[i] * te_parameters[j] + ne_parameters[j] * te_parameters[i]
    ) * ec
    initial_guess[k] = ne_parameters[k] * te_parameters[k] * ec
    # set location guess to be position of maximum gradient in the pressure prediction
    initial_guess[R0_index] = R[1:-1][(pe_prediction[2:] - pe_prediction[:-2]).argmin()]

    # set up a gaussian likelihood function we can maximise to estimate the pressure
    L = GaussianLikelihood(
        y_data=pe_prediction,
        sigma=zeros(n_points) + initial_guess[i] * 0.01,
        forward_model=forward_model,
        forward_model_jacobian=forward_model_jacobian,
    )

    bounds = {
        "pedestal_location": (0.0, None),
        "pedestal_height": (0.0, None),
        "pedestal_width": (1e-3, None),
        "pedestal_top_gradient": (None, None),
        "background_level": (0.0, None),
        "logistic_shape_parameter": (0.05, None),
    }

    result = minimize(
        fun=L.cost,
        x0=initial_guess,
        method="Nelder-Mead",
        bounds=[bounds[p] for p in model.parameters],
        options={"maxiter": 3000},
    )

    if return_diagnostics:
        fit = forward_model(result.x)
        residual = pe_prediction - fit
        diagnostics = {
            "max_abs_err": abs(residual).max(),
            "rmse": sqrt((residual**2).mean()),
            "target": pe_prediction,
            "fit": fit,
            "radius": R,
        }
        return result.x, diagnostics
    else:
        return result.x
