import logging
import numpy as np
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction
from nevergrad.parametrization import core as ng_core

from traits.api import Enum, Unicode, Property

from force_bdss.api import PositiveInt
from force_bdss.mco.optimizer_engines.base_optimizer_engine import (
    BaseOptimizerEngine,
)

log = logging.getLogger(__name__)


class NevergradTypeError(Exception):
    pass


class NevergradOptimizerEngine(BaseOptimizerEngine):

    #: Optimizer name
    name = Unicode("Nevergrad")

    #: Algorithms available to work with
    algorithms = Enum(*ng.optimizers.registry.keys())

    #: Optimization budget defines the allowed number of objective calls
    budget = PositiveInt(500)

    kpi_bounds = Property(depends_on="kpis.[scale_factor]")

    def _algorithms_default(self):
        return "TwoPointsDE"

    @staticmethod
    def _create_instrumentation_variable(parameter):
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

    def _assemble_instrumentation(self, parameters=None):
        """ Assemble nevergrad.Instrumentation object from `parameters` list.

        Parameters
        ----------
        parameters: List(BaseMCOParameter)
            parameter objects containing lower and upper numerical bounds

        Returns
        ----------
        instrumentation: ng.Instrumentation
        """
        if parameters is None:
            parameters = self.parameters

        instrumentation = [
            self._create_instrumentation_variable(p) for p in parameters
        ]
        return ng.p.Instrumentation(*instrumentation)

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

    def optimize(self):
        """ Constructs objects required by the nevergrad engine to
        perform optimization.

        Yields
        ----------
        optimization result: tuple(np.array, np.array, list)
            Point of evaluation, objective value, dummy list of weights
        """
        f = MultiobjectiveFunction(
            multiobjective_function=self._score, upper_bounds=self.kpi_bounds
        )
        instrumentation = self._assemble_instrumentation()
        instrumentation.random_state.seed(12)
        ng_optimizer = ng.optimizers.registry[self.algorithms](
            parametrization=instrumentation, budget=self.budget
        )
        for _ in range(ng_optimizer.budget):
            x = ng_optimizer.ask()
            value = f.multiobjective_function(x.args)
            volume = f.compute_aggregate_loss(
                self._minimization_score(value), *x.args, **x.kwargs
            )
            ng_optimizer.tell(x, volume)

            if self.verbose_run:
                yield x.args, value

        if not self.verbose_run:
            for point, value in f._points:
                value = self._minimization_score(value)
                yield point[0], value
