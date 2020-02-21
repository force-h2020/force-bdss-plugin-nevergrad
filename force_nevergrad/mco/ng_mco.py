import logging

from force_bdss.api import BaseMCO, DataValue

from force_nevergrad.engine.nevergrad_engine import NevergradOptimizerEngine


log = logging.getLogger(__name__)


class NevergradMCO(BaseMCO):
    """ Base Nevergrad MCO class to run gradient-free global optimization.

    The `run` method will perform ".optimize" operation using the
    Nevergrad OptimizerEngine. User should overload this method and
    implement / extend it for custom MCO run.
    """

    def run(self, evaluator):
        model = evaluator.mco_model

        optimizer = NevergradOptimizerEngine(
            kpis=model.kpis,
            parameters=model.parameters,
            budget=model.budget,
            algorithms=model.algorithms,
            single_point_evaluator=evaluator,
            verbose_run=model.verbose_run,
        )

        log.info("Doing MCO run")

        for (optimal_point, optimal_kpis) in optimizer.optimize():
            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            model.notify_progress_event(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
            )
