from traits.api import Enum, Bool
from traitsui.api import View, Item

from force_bdss.api import BaseMCOModel, PositiveInt

from force_nevergrad.engine.nevergrad_engine import NevergradOptimizerEngine


class NevergradMCOModel(BaseMCOModel):

    #: Algorithms available to work with
    algorithms = Enum(
        *NevergradOptimizerEngine.class_traits()["algorithms"].handler.values
    )

    #: Defines the allowed number of objective calls
    budget = PositiveInt(100)

    #: Display the generated points at runtime
    verbose_run = Bool(True)

    def default_traits_view(self):
        return View(
            Item("algorithms"),
            Item("budget", label="Allowed number of objective calls"),
            Item("verbose_run"),
        )

    def _algorithms_default(self):
        return NevergradOptimizerEngine._algorithms_default(None)
