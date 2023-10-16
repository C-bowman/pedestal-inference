from numpy import arange, concatenate, diff, ndarray, zeros


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