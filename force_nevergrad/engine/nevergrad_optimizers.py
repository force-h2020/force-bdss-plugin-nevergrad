#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from functools import partial

import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction
import numpy as np

from traits.api import (
    Enum,
    Float,
    provides,
    HasStrictTraits,
    List
)

from force_bdss.api import (
    PositiveInt,
    IOptimizer
)

from .parameter_translation import (
    translate_mco_to_ng,
    translate_ng_to_mco
)

ALGORITHMS_KEYS = ng.optimizers.registry.keys()


def nevergrad_function(*ng_params,
                       function=None,
                       is_scalar=True):
    """ A wrapper around the MCO objective function,
    that can be optimized by nevergrad.

    Parameters
    ----------
    *ng_params: Any (but usually float, ndarray or string)
        Parameters to be optimized.
    function: Callable
        The MCO objective function (_score(), etc).
    is_scalar: bool
        Whether or not the function should be scalar.
        (be a single-objective function).

    Return
    ------
    float or ndarray
        The objectives/kpis or their sum.

    Notes
    -----
    Nevergrad must optimize a function where each positional or
    keyword argument is a single parameter.
    foo(p1, p2, p3, ...pN)
    MCO objective functions (_score(), etc.) have a single argument -
    the list of parameters.
    foo([p1, p2, p3, ...pN])
    The types of the individual parameters are almost the same, in the
    nevergrad and MCO functions (but see translate_ng_to_mco()).
    Both nevergrad and MCO functions return the objectives/kpis as
    a single, one-dimensional numpy array.
    """

    # pack the nevergrad parameters values
    # into a list of mco parameter values.
    mco_params = translate_ng_to_mco(list(ng_params))

    # call the MCO objective function
    objective = function(mco_params)

    # if the objective needs to be scalar but it is not,
    # sum the objectives into a single objective.
    if is_scalar and not np.isscalar(objective):
        return np.sum(objective)

    return objective


@provides(IOptimizer)
class NevergradScalarOptimizer(HasStrictTraits):
    """ Optimization of a scalar function using nevergrad.
    """

    #: Algorithms available to work with
    algorithms = Enum(*ALGORITHMS_KEYS)

    #: Optimization budget defines the allowed number of objective calls
    budget = PositiveInt(500)

    def _algorithms_default(self):
        return "TwoPointsDE"

    def get_optimizer(self, params):

        instrumentation = translate_mco_to_ng(params)

        return ng.optimizers.registry[self.algorithms](
            parametrization=instrumentation,
            budget=self.budget
        )

    def optimize_function(self, func, params):
        """ Minimize the passed scalar function.

        Parameters
        ----------
        func: Callable
            The MCO function to optimize
            Takes a list of MCO parameter values.
        params: list of MCOParameter
            The MCO parameter objects corresponding to the parameters.

        Yields
        ------
        list of float or list:
            The list of optimal parameter values.
        """
        # Create optimizer.
        optimizer = self.get_optimizer(params)

        # Create a scalar objective Nevergrad function from
        # the MCO function.
        ng_func = partial(nevergrad_function,
                          function=func,
                          is_scalar=True
                          )

        # Optimize.
        # This returns an nevergrad Instrumentation object.
        optimization_result = optimizer.minimize(ng_func)

        # Convert the optimal point into MCO format
        yield translate_ng_to_mco(list(optimization_result.args))


@provides(IOptimizer)
class NevergradMultiOptimizer(HasStrictTraits):
    """ Optimization of a multi-objective function using nevergrad.
    """

    #: Algorithms available to work with
    algorithms = Enum(*ALGORITHMS_KEYS)

    #: Optimization budget defines the allowed number of objective calls
    budget = PositiveInt(500)

    #: List of upper bounds for KPI values
    upper_bounds = List(Float, visible=False, transient=True)

    def _algorithms_default(self):
        return "TwoPointsDE"

    def get_optimizer(self, params):

        instrumentation = translate_mco_to_ng(params)

        return ng.optimizers.registry[self.algorithms](
            parametrization=instrumentation,
            budget=self.budget
        )

    def get_multiobjective_function(self, ng_func):

        return MultiobjectiveFunction(multiobjective_function=ng_func)

    def optimize_function(self, func, params, verbose_run=False):
        """ Minimize the passed multi-objective function.

        Parameters
        ----------
        func: Callable
            The MCO function to optimize
            Takes a list of MCO parameter values.
        params: list of MCOParameter
            The MCO parameter objects corresponding to the parameters.
        verbose_run: Bool, optional
            Whether or not to return all points generated during the
            optimization procedure, or just those on the Pareto front.

        Yields
        ------
        list of float or list:
            The list of parameter values for a single member
            of the Pareto set.
        """

        # Create optimizer.
        optimizer = self.get_optimizer(params)

        # Create a multi-objective nevergrad function from
        # the MCO function.
        ng_func = partial(nevergrad_function,
                          function=func,
                          is_scalar=False
                          )

        # Create a MultiobjectiveFunction object from that.
        # Once we have defined an upper_bound attribute for KPIs we can
        # then pass these (as a numpy array) to the upper_bounds argument
        # of MultiobjectiveFunction.
        # upper_bounds=np.array([k.upper_bound for k in self.kpis])
        ob_func = self.get_multiobjective_function(ng_func)

        # Optimize. Ignore the return.
        optimizer.minimize(ob_func)

        # yield a member of the Pareto set.
        # x is a tuple - ((<vargs parameters>), {<kwargs parameters>})
        # return the vargs, translated into mco.
        for x in ob_func.pareto_front():
            yield translate_ng_to_mco(list(x[0]))
