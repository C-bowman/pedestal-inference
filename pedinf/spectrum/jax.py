from numpy import ndarray
import jax.numpy as jnp
from jax import jit


def response(
    ln_te: ndarray,
    angle_diffs: ndarray,
    y: ndarray,
    a: ndarray,
    b: ndarray,
    ln_te_knots: ndarray,
    knot_spacing: float,
):
    # get the spline coordinate
    inds = jnp.searchsorted(ln_te_knots, ln_te) - 1
    t = (ln_te - ln_te_knots[inds]) / knot_spacing
    u = 1 - t
    ut = u * t

    y_u = jnp.take_along_axis(y, inds[:, None, :, None], axis=2)
    y_t = jnp.take_along_axis(y, inds[:, None, :, None] + 1, axis=2)
    a_slc = jnp.take_along_axis(a, inds[:, None, :, None], axis=2)
    b_slc = jnp.take_along_axis(b, inds[:, None, :, None], axis=2)

    t1 = u[:, None, :, None] * y_u
    t2 = t[:, None, :, None] * y_t
    t3 = u[:, None, :, None] * a_slc
    t4 = t[:, None, :, None] * b_slc
    splines = t1 + t2 + (t3 + t4) * ut[:, None, :, None]
    return splines[:, :, :, 0] + angle_diffs[:, None, :] * splines[:, :, :, 1]


def response_and_grad(
    ln_te: ndarray,
    angle_diffs: ndarray,
    y: ndarray,
    a: ndarray,
    b: ndarray,
    ln_te_knots: ndarray,
    knot_spacing: float,
):
    # get the spline coordinate
    inds = jnp.searchsorted(ln_te_knots, ln_te) - 1
    t = (ln_te - ln_te_knots[inds]) / knot_spacing
    u = 1 - t
    ut = u * t

    y_u = jnp.take_along_axis(y, inds[:, None, :, None], axis=2)
    y_t = jnp.take_along_axis(y, inds[:, None, :, None] + 1, axis=2)
    a_slc = jnp.take_along_axis(a, inds[:, None, :, None], axis=2)
    b_slc = jnp.take_along_axis(b, inds[:, None, :, None], axis=2)

    t1 = u[:, None, :, None] * y_u
    t2 = t[:, None, :, None] * y_t
    t3 = u[:, None, :, None] * a_slc
    t4 = t[:, None, :, None] * b_slc
    splines = t1 + t2 + (t3 + t4) * ut[:, None, :, None]
    return (
        splines[:, :, :, 2] + angle_diffs[:, None, :] * splines[:, :, :, 3],
        splines[:, :, :, 0] + angle_diffs[:, None, :] * splines[:, :, :, 1],
    )


@jit
def spectrum(
    Te: ndarray,
    ne: ndarray,
    angle_diffs: ndarray,
    weights: ndarray,
    y: ndarray,
    a: ndarray,
    b: ndarray,
    ln_te_knots: ndarray,
    knot_spacing: float,
):
    ln_te = jnp.log(Te)
    coeffs = ne * weights
    res = response(
        ln_te, angle_diffs, y, a, b, ln_te_knots, knot_spacing,
    )
    return jnp.sum(res * coeffs[:, None, :], axis=2)


@jit
def spectrum_jacobian(
    Te: ndarray,
    ne: ndarray,
    angle_diffs: ndarray,
    weights: ndarray,
    y: ndarray,
    a: ndarray,
    b: ndarray,
    ln_te_knots: ndarray,
    knot_spacing: float,
):
    ln_te = jnp.log(Te)
    gradient, res = response_and_grad(
        ln_te, angle_diffs, y, a, b, ln_te_knots, knot_spacing,
    )
    coeffs = (ne * weights) / Te
    return gradient * coeffs[:, None, :], res * weights[:, None, :]
