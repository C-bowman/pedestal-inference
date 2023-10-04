from numpy import arange, concatenate, exp, ndarray, zeros, diff
from pedinf.models import ProfileModel


class logispline(ProfileModel):
    name = "logispline"

    def __init__(self, knots: ndarray, radius=None, low_field_side=True):
        self.knots = knots
        if radius is not None:
            self.radius = radius
            self.basis = b_spline_basis(self.radius, self.knots)
        self.drn = -1 if low_field_side else 1

        self.n_parameters = 3 + self.knots.size
        self.parameters = {
            "pedestal_location": 0,
            "pedestal_height": 1,
            "pedestal_width": 2,
            "basis_weights": slice(3, self.n_parameters)
        }

    def prediction(self, R: ndarray, theta: ndarray) -> ndarray:
        basis = b_spline_basis(R, self.knots)
        z = (4 * self.drn) * (R - theta[0]) / theta[2]
        logistic = theta[1] / (1 + exp(-z))
        background = basis @ theta[3:]
        return logistic + background

    def forward_prediction(self, theta: ndarray) -> ndarray:
        z = (4 * self.drn) * (self.radius - theta[0]) / theta[2]
        logistic = theta[1] / (1 + exp(-z))
        background = self.basis @ theta[3:]
        return logistic + background

    def jacobian(self, R: ndarray, theta: ndarray):
        basis = b_spline_basis(R, self.knots)
        z = (4 * self.drn) * (R - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - L)

        jac = zeros([R.size, self.n_parameters])
        jac[:, 0] = -q * ((4 * self.drn * theta[1]) / theta[2])
        jac[:, 1] = L
        jac[:, 2] = -q * z * (theta[1] / theta[2])
        jac[:, self.parameters["basis_weights"]] = basis
        return jac

    def prediction_and_jacobian(self, R: ndarray, theta: ndarray):
        basis = b_spline_basis(R, self.knots)
        z = (4 * self.drn) * (R - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - L)

        prediction = theta[1] * L + basis @ theta[3:]

        jac = zeros([R.size, self.n_parameters])
        jac[:, 0] = -q * ((4 * self.drn * theta[1]) / theta[2])
        jac[:, 1] = L
        jac[:, 2] = -q * z * (theta[1] / theta[2])
        jac[:, self.parameters["basis_weights"]] = basis
        return prediction, jac

    def gradient(self, R: ndarray, theta: ndarray):
        _, derivs = b_spline_basis(R, self.knots, derivatives=True)
        z = (4 * self.drn) * (R - theta[0]) / theta[2]
        L = 1 / (1 + exp(-z))
        q = L * (1 - L)
        return q * (4 * self.drn * theta[1] / theta[2]) + derivs @ theta[3:]


def b_spline_basis(x: ndarray, knots: ndarray, order=3, derivatives=False) -> ndarray:
    assert order % 2 == 1
    iters = order + 1
    t = knots.copy()
    # we need n = order points of padding either side
    dl, dr = t[1] - t[0], t[-1] - t[-2]
    L_pad = t[0] - dl * arange(1, iters)[::-1]
    R_pad = t[-1] + dr * arange(1, iters)

    t = concatenate([L_pad, t, R_pad])
    n_knots = t.size

    # construct zeroth-order
    splines = zeros([x.size, n_knots, iters])
    for i in range(n_knots - 1):
        bools = (t[i] <= x) & (t[i + 1] > x)
        splines[bools, i, 0] = 1.

    dx = x[:, None] - t[None, :]
    for k in range(1, iters):
        dt = t[k:] - t[:-k]
        S = splines[:, :-k, k-1] / dt[None, :]
        splines[:, :-(k+1), k] = S[:, :-1] * dx[:, :-(k+1)] - S[:, 1:] * dx[:, k+1:]

    # remove the excess functions which don't contribute to the supported range
    basis = splines[:, :-iters, -1]

    # combine the functions at the edge of the supported range
    if iters // 2 > 1:
        n = (iters // 2) - 1
        basis[:, n] += basis[:, :n].sum(axis=1)
        basis[:, -(n+1)] += basis[:, -n:].sum(axis=1)
        basis = basis[:, n:-n]

    if derivatives:
        # derivative of order k splines are a weighted difference of order k-1 splines
        coeffs = order / (t[:-order] - t[order:])
        derivs = diff(splines[:, :-order, -2] * coeffs[None, :])

        if iters // 2 > 1:
            derivs[:, n] += derivs[:, :n].sum(axis=1)
            derivs[:, -(n+1)] += derivs[:, -n:].sum(axis=1)
            derivs = derivs[:, n:-n]

        return basis, derivs
    else:
        return basis