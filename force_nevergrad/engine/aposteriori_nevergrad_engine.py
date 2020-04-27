
from force_bdss.mco.optimizer_engines.aposteriori_optimizer_engine import (
    AposterioriOptimizerEngine
)
from force_bdss.mco.optimizer_engines.weighted_optimizer_engine import (
    WeightedOptimizerEngine
)
from force_bdss.mco.optimizers.nevergrad_optimizers import (
    NevergradMultiOptimizer,
    NevergradScalarOptimizer
)


class AposterioriNevergradEngine(NevergradMultiOptimizer,
                                 AposterioriOptimizerEngine):
    """ A posteriori multi-objective optimization
    using the nevergrad library.
    """
    pass


class AprioriNevergradEngine(NevergradScalarOptimizer,
                             WeightedOptimizerEngine):
    """ A priori (weighted) multi-objective optimization
    using the nevergrad library.
    Just here to illustrate the point.
    """
    pass
