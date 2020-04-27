
from force_bdss.mco.optimizer_engines.aposteriori_optimizer_engine import (
    AposterioriOptimizerEngine
)
from force_bdss.mco.optimizers.nevergrad_optimizers import (
    NevergradMultiOptimizer,
    NevergradScalarOptimizer
)


class AposterioriNevergradEngine(NevergradMultiOptimizer,
                                 AposterioriOptimizerEngine):
    pass


class AprioriNevergradEngine(NevergradScalarOptimizer,
                             AposterioriOptimizerEngine):
    pass
