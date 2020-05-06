import numpy as np
from functools import partial

from traits.api import (
    Enum,
    provides,
    HasStrictTraits,
    List
)

from force_bdss.api import (
    PositiveInt,
    KPISpecification,
    IOptimizer
)

from .parameter_translation import (
    translate_mco_to_ng,
    translate_ng_to_mco
)
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction

ALGORITHMS_KEYS = ng.optimizers.registry.keys()


def nevergrad_function(*ng_params,
                       function=None,
                       is_scalar=True,
                       minimize_objectives=[]):
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
    minimize_objectives: list of bool
        For each objective, whether it should be minimized
        (as opposed to maximized).

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

    # negate any objectives that should be maximised
    if len(minimize_objectives) == len(objective):
        for i, do_minimize in enumerate(minimize_objectives):
            if not do_minimize:
                objective[i] = -objective[i]

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

    #: A list of the output KPI parameters representing the objective(s)
    kpis = List(KPISpecification, visible=False, transient=True)
    # ...probably kpis should be passed as an arg to optimize_function()
    # in future implementations. kpis are not a property of the optimizer!

    def _algorithms_default(self):
        return "TwoPointsDE"

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
        # Create instrumentation.
        instrumentation = translate_mco_to_ng(params)

        # Create optimizer.
        optimizer = ng.optimizers.registry[self.algorithms](
            parametrization=instrumentation,
            budget=self.budget
        )

        # Minimization/maximization specification
        minimize_objectives = ['MINI' in k.objective for k in self.kpis]

        # Create a scalar objective nevergrad function from
        # the MCO function.
        ng_func = partial(nevergrad_function,
                          function=func,
                          is_scalar=True,
                          minimize_objectives=minimize_objectives
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

    #: A list of the output KPI parameters representing the objective(s)
    kpis = List(KPISpecification, visible=False, transient=True)

    def _algorithms_default(self):
        return "TwoPointsDE"

    def optimize_function(self, func, params):
        """ Minimize the passed multi-objective function.

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
            The list of parameter values for a single member
            of the Pareto set.
        """

        # Create instrumentation.
        instrumentation = translate_mco_to_ng(params)

        # Create optimizer.
        optimizer = ng.optimizers.registry[self.algorithms](
            parametrization=instrumentation,
            budget=self.budget
        )

        # Minimization/maximization specification
        minimize_objectives = ['MINI' in k.objective for k in self.kpis]

        # Create a multi-objective nevergrad function from
        # the MCO function.
        ng_func = partial(nevergrad_function,
                          function=func,
                          is_scalar=False,
                          minimize_objectives=minimize_objectives
                          )

        # Create a MultiobjectiveFunction object from that.
        # Once we have defined an upper_bound attribute for KPIs we can
        # then pass these (as a numpy array) to the upper_bounds argument
        # of MultiobjectiveFunction.
        # upper_bounds=np.array([k.upper_bound for k in self.kpis])
        ob_func = MultiobjectiveFunction(multiobjective_function=ng_func)

        # Optimize. Ignore the return.
        optimizer.minimize(ob_func)

        # yield a member of the Pareto set.
        # x is a tuple - ((<vargs parameters>), {<kwargs parameters>})
        # return the vargs, translated into mco.
        for x in ob_func.pareto_front():
            yield translate_ng_to_mco(list(x[0]))
