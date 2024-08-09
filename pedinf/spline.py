from numpy import ndarray, zeros, searchsorted
from scipy.linalg import solveh_banded


def cubic_spline_coefficients(x: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
    """
    Computes the coefficients of a natural cubic spline in 'symmetric' form.
    """
    assert x.size == y.size
    assert x.ndim == y.ndim == 1
    assert x.size > 2
    A = zeros([2, x.size])
    b = zeros([x.size])
    n = x.size - 1

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    assert (dx > 0).all()

    v = 1 / dx
    g = dy * v**2

    # build matrix diagonal
    A[1, 0] = 2 * v[0]
    A[1, 1:-1] = 2 * (v[1:] + v[:-1])
    A[1, n] = 2 * v[-1]

    # build off-diagonals
    A[0, 1:] = v

    # build linear system target values
    b[1:-1] = 3 * (g[:-1] + g[1:])
    b[0] = 3 * g[0]
    b[n] = 3 * g[-1]

    # solve the system to get the cubic coefficients
    k = solveh_banded(A, b)
    a_coeffs = k[:-1] * dx - dy
    b_coeffs = -k[1:] * dx + dy
    return a_coeffs, b_coeffs


def evaluate_cubic_spline(
    x: ndarray, x_knots: ndarray, y_knots: ndarray, a: ndarray, b: ndarray
) -> ndarray:
    inds = searchsorted(x_knots, x) - 1
    dk = x_knots[1:] - x_knots[:-1]
    t = (x - x_knots[inds]) / dk[inds]
    u = 1 - t
    return y_knots[inds] * u + y_knots[inds + 1] * t + u * t * (u * a[inds] + t * b[inds])
