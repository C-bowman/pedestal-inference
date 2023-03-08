from numpy import array
from pedinf.models import lpm
from pedinf.analysis import locate_radius

def test_locate_radius():
    theta = array([1.4, 150, 0.02, 600, 5, -0.5])
    for value in [30., 40., 50., 60.]:
        R = locate_radius(profile_value=value, theta=theta, model=lpm)
        assert abs(lpm.prediction(R, theta) / value - 1.).max() < 1e-8