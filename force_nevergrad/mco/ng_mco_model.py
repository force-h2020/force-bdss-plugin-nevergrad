from traits.api import Enum, Bool
from traitsui.api import View, Item

from force_bdss.api import BaseMCOModel, PositiveInt

from force_nevergrad.engine.nevergrad_optimizers import ALGORITHMS_KEYS


class NevergradMCOModel(BaseMCOModel):
    """ Base NevergradMCO Model class. Contains necessary traits attribute
    data to configure the NevergradOptimizerEngine."""

    #: Algorithms available to work with
    algorithms = Enum(*ALGORITHMS_KEYS)

    #: Defines the allowed number of objective calls
    budget = PositiveInt(100)

    #: Display the generated points at runtime
    verbose_run = Bool(True)

    def _algorithms_default(self):
        return "TwoPointsDE"

    def default_traits_view(self):
        return View(
            Item("algorithms"),
            Item("budget", label="Allowed number of objective calls"),
            Item("verbose_run"),
        )
