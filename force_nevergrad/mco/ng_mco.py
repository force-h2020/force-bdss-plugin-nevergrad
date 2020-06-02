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


class NevergradMCO(BaseMCO):
    """ Base Nevergrad MCO class to run gradient-free global optimization.

    The `run` method will perform ".optimize" operation using the
    Nevergrad OptimizerEngine. User should overload this method and
    implement / extend it for custom MCO run.
    """

    def run(self, evaluator):
        model = evaluator.mco_model

        optimizer = NevergradMultiOptimizer(
            algorithms=model.algorithms,
            kpis=model.kpis,
            budget=model.budget)

        engine = AposterioriOptimizerEngine(
            kpis=model.kpis,
            parameters=model.parameters,
            single_point_evaluator=evaluator,
            verbose_run=model.verbose_run,
            optimizer=optimizer
        )

        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        log.addHandler(screen_handler)

        for index, (optimal_point, optimal_kpis) \
                in enumerate(engine.optimize()):
            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            log.info("Doing  MCO run # {}".format(index))
            model.notify_progress_event(
                [DataValue(value=v) for v in optimal_point],
                [DataValue(value=v) for v in optimal_kpis],
            )
