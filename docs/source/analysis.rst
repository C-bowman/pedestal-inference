Analysis of edge profile inference results
==========================================

The ``pedinf.analysis`` module provides tools for analysis and post-processing
of inferred electron temperature and density profiles.


Inferring the separatrix given a temperature
--------------------------------------------

If the separatrix temperature has been estimated, for example
via power-balance analysis, the ``separatrix_given_temperature``
function can be used to infer the separatrix radius, density
and pressure.

.. autofunction:: pedinf.analysis.separatrix_given_temperature


Inferring the separatrix given a scaling
----------------------------------------

Given a scaling which predicts the separatrix electron temperature
as a function of the separatrix electron density (e.g. from modelling),
the ``separatrix_given_scaling`` function can be used to infer the
separatrix radius, temperature, density and pressure.


.. autofunction:: pedinf.analysis.separatrix_given_scaling


Inferring the pressure profile
------------------------------

The ``pressure_profile_and_gradient`` function generates a sample
of pressure and pressure gradient profiles and associated means and
highest-density intervals.

.. autofunction:: pedinf.analysis.pressure_profile_and_gradient


Estimating pressure profile parameters
--------------------------------------

The ``pressure_parameters`` function estimates the parameters of the electron
pressure for a given edge profile model and corresponding set of temperature
and density parameters.

.. autofunction:: pedinf.analysis.pressure_parameters