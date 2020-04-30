import numpy as np
from functools import partial
from traits.api import (
    Enum,
    provides,
    HasStrictTraits,
    Property,
    List
)
from force_bdss.api import PositiveInt

from force_bdss.core.kpi_specification import KPISpecification

from force_bdss.mco.optimizers.i_optimizer import IOptimizer

import nevergrad as ng
from nevergrad.parametrization import core as ng_core
from nevergrad.functions import MultiobjectiveFunction

import logging
log = logging.getLogger(__name__)


class NevergradTypeError(Exception):
    pass


def create_instrumentation_variable(parameter):
    """ Create nevergrad.variable from `MCOParameter`. Different
    MCOParameter subclasses have different signature attributes.
    The mapping between MCOParameters and nevergrad types is bijective.

    Parameters
    ----------
    parameter: BaseMCOParameter
        object to convert to nevergrad type

    Returns
    ----------
    nevergrad.Variable
        nevergrad variable of corresponding type
    """
    if hasattr(parameter, "lower_bound") and hasattr(
            parameter, "upper_bound"
    ):
        mid_point = parameter.initial_value
        try:
            _ = parameter.dimension
        except AttributeError:
            parameter_type = ng.p.Scalar
        else:
            parameter_type = ng.p.Array
        return parameter_type(init=mid_point).set_bounds(
            parameter.lower_bound, parameter.upper_bound, method="arctan"
        )
    elif hasattr(parameter, "value"):
        return ng_core.Constant(value=parameter.value)
    elif hasattr(parameter, "levels"):
        return ng.p.TransitionChoice(parameter.sample_values)
    elif hasattr(parameter, "categories"):
        return ng.p.Choice(
            choices=parameter.sample_values, deterministic=True
        )
    else:
        raise NevergradTypeError(
            f"Can not convert {parameter} to any of"
            " supported Nevergrad types"
        )


@provides(IOptimizer)
class NevergradScalarOptimizer(HasStrictTraits):
    """ Optimization of a scalar function using nevergrad.
    """

    #: Algorithms available to work with
    algorithms = Enum(*ng.optimizers.registry.keys())

    #: Optimization budget defines the allowed number of objective calls
    budget = PositiveInt(500)

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
                    The MCO parameter objects corresponding to the parameter values.

                Yields
                ------
                list of float or list:
                    The list of optimal parameter values.
        """
        # Create parameterization.
        instrumentation = ng.p.Instrumentation(
            *[create_instrumentation_variable(p) for p in params]
        )

        # Create optimizer.
        optimizer = ng.optimizers.registry[self.algorithms](
            parametrization=instrumentation, budget=self.budget
        )

        # Optimize.
        optimization_result = optimizer.minimize(func)
        optimal_point = optimization_result.value

        yield optimal_point


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

    kpi_bounds = Property(depends_on="kpis.[scale_factor]")

    def _algorithms_default(self):
        return "TwoPointsDE"

    def _get_kpi_bounds(self):
        """ Assemble optimization bounds on KPIs, provided by
        the `scale_factor` attributes.
        Note: Ideally, an `upper_bound` kpi attribute should be
        responsible for the bounds.
        Parameters
        ----------
        kpis: List(KPISpecification)
            kpi objects containing upper numerical bounds
        Returns
        ----------
        upper_bounds: np.array
            kpis upper bounds
        """
        upper_bounds = np.zeros(len(self.kpis))
        for i, kpi in enumerate(self.kpis):
            upper_bounds[i] = kpi.scale_factor
        return upper_bounds

    def optimize_function(self, func, params):
        """ Minimize the passed multi-objective function.

                Parameters
                ----------
                func: Callable
                    The MCO function to optimize
                    Takes a list of MCO parameter values.
                params: list of MCOParameter
                    The MCO parameter objects corresponding to the parameter values.

                Yields
                ------
                list of float or list:
                    The list of parameter optimal values.
        """
        # create partial of func, where parameter args are unpacked
        ufunc = partial(self.unpacked_func, func=func)

        # create multi-objective function object
        mfunc = MultiobjectiveFunction(
            multiobjective_function=ufunc, upper_bounds=self.kpi_bounds
        )

        # Create parameterization.
        instrumentation = ng.p.Instrumentation(
           *[create_instrumentation_variable(x) for x in params]
        )
        #print(str(instrumentation))

        # Create optimizer.
        # NOTE: parametrization = instrumentation, DOES NOT WORK
        # for multi-objective function args should be unpacked?
        optimizer = ng.optimizers.registry[self.algorithms](
            parametrization=instrumentation, budget=self.budget
        )

        # Optimize.
        optimizer.minimize(mfunc)

        # yield a member of the Pareto set.
        for x in mfunc.pareto_front():
            print(type(x[0]))
            yield x[0][0]

    def unpacked_func(self, *vargs, func=None):

        kp = func(list(vargs))
        #print('output = ', type(kp), len(kp))
        return kp


