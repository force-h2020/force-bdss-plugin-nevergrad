import numpy as np
from functools import partial

from traits.api import (
    Enum,
    provides,
    HasStrictTraits,
    List
)
from force_bdss.mco.parameters.mco_parameters import (
    FixedMCOParameter,
    RangedMCOParameter,
    RangedVectorMCOParameter,
    ListedMCOParameter,
    CategoricalMCOParameter,
)

from force_bdss.api import PositiveInt
from force_bdss.core.kpi_specification import KPISpecification
from force_bdss.mco.optimizers.i_optimizer import IOptimizer

import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction


def translate_mco_to_ng(params):
    r""" Translate from an MCO parameter specification
    to a Nevergrad parameter specification (Instrumentation).

    Parameters
    ----------
    params: list of MCOParameter
        The MCO parameter specification.

    Return
    ------
    Instrumentation
        Nevergrad instrumentation object.

    Notes
    -----
    The Instrumentation object can be created from a list and/or
    a dict of nevergrad parameter types. The keys of the dict are the
    names of the parameters.
    These list/dict is passed to the Instrumentation __init__ as
    \*vargs/\*kwargs and they set the attributes called args/kwargs.
    The result of a nevergrad optimization is itself a Instrumentation object.

    As MCO objective functions (_score(), etc) take their parameters as a
    single list argument, rather than kwargs (i.e. parameters are indexed
    by position) we must create the Instrumentation by \*vargs. Also against
    using \*kwargs, is that MCO parameter names may not be unique (there is
    nothing to enforce this).
    """

    instru = []
    for p in params:

        ng_param = None
        if isinstance(p, FixedMCOParameter):
            ng_param = ng.p.Constant(
                value=p.value,
            )
        elif isinstance(p, RangedVectorMCOParameter):
            ng_param = ng.p.Array(
                init=np.array(p.initial_value),
                mutable_sigma=True
            )
            ng_param.set_bounds(
                lower=np.array(p.lower_bound),
                upper=np.array(p.upper_bound),
                method="constraint"
            )
        elif isinstance(p, RangedMCOParameter):
            ng_param = ng.p.Scalar(
                init=p.initial_value,
                lower=p.lower_bound,
                upper=p.upper_bound,
                mutable_sigma=False
            )
        elif isinstance(p, ListedMCOParameter):
            ng_param = ng.p.TransitionChoice(
                choices=p.levels,
                transitions=[1.0, 1.0]
            )
        elif isinstance(p, CategoricalMCOParameter):
            ng_param = ng.p.Choice(
                choices=p.categories,
                deterministic=False
            )
        else:
            # duck-typing for non-standard??? quack, quack
            ng_param = ng.p.Constant(
                value="null"
            )

        # add parameter to list
        instru.append(ng_param)

    # create Instrumentation object with *vargs
    return ng.p.Instrumentation(*instru)


def translate_ng_to_mco(ng_params):
    """ Translate a list of nevergrad parameter values
    to a list of MCO parameter values.

    Parameters
    ----------
    ng_params: list of Any (but usually float, ndarray or string)
        Parameter values in the nevergrad form

    Return
    ------
    mco_values: list of Any (but usually float, list or string)
        Parameter values in the MCO form

    Notes
    -----
    These are mostly the same, except for RangedVectorMCOParameter to
    ng.p.Array conversion: the value of the former is a list, whereas
    the value of the latter is a numpy array
    """

    mco_values = []
    for p in ng_params:
        if isinstance(p, np.ndarray):
            mco_values.append(p.tolist())
        else:
            mco_values.append(p)
        # ...what about any non-standard MCOParameter types?
        # (see translate_mco_to_ng(), above)

    return mco_values


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
    if minimize_objectives:
        for i, do_minimize in enumerate(minimize_objectives):
            if (not do_minimize) and (i < len(objective)):
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
    algorithms = Enum(*ng.optimizers.registry.keys())

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
    algorithms = Enum(*ng.optimizers.registry.keys())

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
        ob_func = MultiobjectiveFunction(
            multiobjective_function=ng_func,
            upper_bounds=[k.scale_factor for k in self.kpis]
        )

        # Optimize. Ignore the return.
        optimizer.minimize(ob_func)

        # yield a member of the Pareto set.
        # x is a tuple - ((<vargs parameters>), {<kwargs parameters>})
        # return the vargs, translated into mco.
        for x in ob_func.pareto_front():
            yield translate_ng_to_mco(list(x[0]))
