#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

import logging
import sys

from force_bdss.api import BaseMCO, DataValue

from force_bdss.mco.optimizer_engines.aposteriori_optimizer_engine import (
    AposterioriOptimizerEngine
)
from force_nevergrad.engine.nevergrad_optimizers import (
    NevergradMultiOptimizer
)

log = logging.getLogger(__name__)


class NevergradOptimizerEngine(AposterioriOptimizerEngine):
    """Extends the AposterioriOptimizerEngine class to include
    transformation of raw KPI bounds into minimization score
    upper bounds
    """

    def score_upper_bounds(self):
        """Returns KPI upper_bounds that have been transformed
        into upper bounds of the minimization score function
        """

        # If KPI objective is to maximise the value, then use
        # the lower bound attribute. Otherwise use the upper
        # bound attribute
        kpi_bounds = [
            kpi.lower_bound
            if kpi.objective == 'MAXIMISE' else kpi.upper_bound
            for kpi in self.kpis
        ]

        # Transform the raw KPI bounds to the minimization score
        score_upper_bounds = self._minimization_score(kpi_bounds)

        # Return only those values that correspond to KPIs with
        # the use_bounds attribute set to True
        return [
            value if kpi.use_bounds else None
            for kpi, value in zip(self.kpis, score_upper_bounds)]


class NevergradMCO(BaseMCO):
    """ Base Nevergrad MCO class to run gradient-free global optimization.

    The `run` method will perform ".optimize" operation using the
    Nevergrad OptimizerEngine. User should overload this method and
    implement / extend it for custom MCO run.
    """

    def run(self, evaluator):
        model = evaluator.mco_model

        engine = NevergradOptimizerEngine(
            kpis=model.kpis,
            parameters=model.parameters,
            single_point_evaluator=evaluator,
            verbose_run=model.verbose_run
        )

        # Transform the KPI upper bounds values using the
        # score function
        upper_bounds = engine.score_upper_bounds()

        # Assign optimizer with KPI score upper bounds
        engine.optimizer = NevergradMultiOptimizer(
            algorithms=model.algorithms,
            budget=model.budget,
            upper_bounds=upper_bounds
        )

        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        log.addHandler(screen_handler)

        for index, (optimal_point, optimal_kpis) \
                in enumerate(engine.optimize(verbose_run=model.verbose_run)):
            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            log.info("Doing  MCO run # {}".format(index))
            model.notify_progress_event(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
            )
