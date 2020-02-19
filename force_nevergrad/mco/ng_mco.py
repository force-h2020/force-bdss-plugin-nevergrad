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

        label_dict = get_labels(model.parameters)

        log.info("Doing MCO run")

        for (
            optimal_point,
            optimal_kpis,
        ) in optimizer.optimize():
            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            readable_points = [
                label_dict[v] if v in label_dict else v for v in optimal_point
            ]
            model.notify_progress_event(
                [DataValue(value=v) for v in readable_points],
                [DataValue(value=v) for v in optimal_kpis],
            )


def get_labels(parameters):
    """ Generates numerical labels for each categorical MCOParameter"""

    label_dict = {}
    label = 1

    for parameter in parameters:
        if hasattr(parameter, "categories"):
            for name in parameter.categories:
                if name not in label_dict:
                    label_dict[name] = label
                    label += 1

    return label_dict
