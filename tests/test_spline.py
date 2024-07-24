from numpy import exp, log, sin, linspace
from pedinf.spline import cubic_spline_coefficients, evaluate_cubic_spline


def sinexp(x):
    return sin(3*x) * exp(-x)


def test_cubic_spline():
    x_knots = exp(linspace(log(1), log(6), 64))
    x_test = 0.5 * (x_knots[1:] + x_knots[:-1])
    y_knots = sinexp(x_knots)
    y_test = sinexp(x_test)

    a_coeff, b_coeff = cubic_spline_coefficients(x_knots, y_knots)
    y_spline = evaluate_cubic_spline(x_test, x_knots, y_knots, a_coeff, b_coeff)

    max_error = abs(y_spline - y_test).max()
    assert max_error < 1e-4
