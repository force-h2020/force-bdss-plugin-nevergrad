#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from functools import partial
import logging

import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction
import numpy as np

from traits.api import (
    Enum,
    Float,
    provides,
    HasStrictTraits,
    List,
    Union
)

from force_bdss.api import (
    PositiveInt,
    IOptimizer
)

from .parameter_translation import (
    translate_mco_to_ng,
    translate_ng_to_mco
)


log = logging.getLogger(__name__)

ALGORITHMS_KEYS = ng.optimizers.registry.keys()


def _nevergrad_ask_tell(optimizer, ob_func, no_bias=False):
    """Exposes the Nevergrad Optimizer ask and tell interface

    Parameters
    ----------
    optimizer: nevergrad.Optimizer
        Nevergrad Optimizer instance to perform optimization routine
    ob_func: nevergrad.MultiobjectiveFunction
        Nevergrad MultiobjectiveFunction instance to be optimized
    no_bias: bool, optional
        Whether or nor to calculate hyper-volume from objective
        function value and return to optimizer

    Returns
    -------
    x: nevergrad.Parameter
        Parameter values determining input point to be calculated
    value: float
        Output value calculated from objective function
    """

    # Ask the optimizer for a new value
    x = optimizer.ask()

    # Calculate the optimizer objective score values
    value = ob_func.multiobjective_function(*x.args)

    # Update the objective function with the new value and
    # compute the hyper-volume. If no_bias is enforced, then
    # do not report any information to both optimizer or
    # objective function
    if no_bias:
        volume = 0
    else:
        volume = ob_func.compute_aggregate_loss(
            value, *x.args, **x.kwargs
        )

    # Tell hyper-volume information to the optimizer
    optimizer.tell(x, volume)

    # Return reference to both input and output values
    return x, value


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

    #: Defines the sample size to estimate the KPI upper bounds
    bound_sample = PositiveInt(15)

    #: List of upper bounds for KPI values
    upper_bounds = List(Union(None, Float), visible=False, transient=True)

    def _algorithms_default(self):
        return "TwoPointsDE"

    def _valid_upper_bounds(self):
        """Returns whether or not the KPI upper bounds need to be
        estimated prior to running the optimization proceedure.
        """
        # If no upper_bounds have been set, we need to estimate
        if len(self.upper_bounds) == 0:
            return False

        # If any upper_bound values are not defined,
        # we need to estimate
        return all([value is not None for value in self.upper_bounds])

    def _calculate_upper_bounds(self, optimizer, function):
        """Uses Nevergrad's MultiobjectiveFunction.compute_aggregate_loss
        protocol to estimate the upper bounds of each output KPI. This
        is only needed if we have a mixture of KPIs that use bounds and
        do not use bounds.
        """

        ob_func = MultiobjectiveFunction(
            multiobjective_function=function)

        # Prior estimate of upper_bounds ensures the calculated KPIs
        # are always higher
        upper_bounds = np.array([-np.inf])

        # Calculate a small random sample of output KPI scores
        for _ in range(self.bound_sample):
            # Use the optimizer to generate a new input / output point
            x, value = _nevergrad_ask_tell(
                optimizer, ob_func, no_bias=True)

            # Keep track of the highest bound
            upper_bounds = np.maximum(upper_bounds, value)

        # And replace those not defined
        return [
            estimate if bound is None else bound
            for estimate, bound in zip(upper_bounds, self.upper_bounds)
        ]

    def get_optimizer(self, params):

        instrumentation = translate_mco_to_ng(params)
        return ng.optimizers.registry[self.algorithms](
            parametrization=instrumentation,
            budget=self.budget
        )

    def get_multiobjective_function(self, ng_func, upper_bounds=None):
        return MultiobjectiveFunction(
            multiobjective_function=ng_func,
            upper_bounds=upper_bounds
        )

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

        # If a complete set of KPI upper bounds are defined, use them.
        # Otherwise use Nevergrad to estimate those not defined
        if self._valid_upper_bounds():
            upper_bounds = self.upper_bounds
        else:
            # Estimate all KPI upper bounds
            upper_bounds = self._calculate_upper_bounds(optimizer, ng_func)

        # Create a MultiobjectiveFunction object with assigned upper bounds
        ob_func = self.get_multiobjective_function(ng_func, upper_bounds)

        # Perform all calculations in the budget
        for index in range(self.budget):
            log.info("Doing  MCO run # {} / {}".format(index, self.budget))

            # Generate and solve a new input point
            x, _ = _nevergrad_ask_tell(optimizer, ob_func)

            # If verbose, report back all points, not just those in
            # Pareto front
            if verbose_run:
                yield translate_ng_to_mco(x.args)

        # If not verbose, yield each member of the Pareto set.
        # x is a tuple - ((<vargs parameters>), {<kwargs parameters>})
        # return the vargs, translated into mco.
        if not verbose_run:
            for x in ob_func.pareto_front():
                yield translate_ng_to_mco(list(x[0]))
