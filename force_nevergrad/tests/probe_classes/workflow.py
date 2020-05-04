from traits.api import Instance, provides

from force_bdss.api import IEvaluator

from force_nevergrad.tests.probe_classes.optimizer import (
    GridValleyObjective
)

from force_nevergrad.mco.ng_mco_factory import NevergradMCOModel


@provides(IEvaluator)
class ProbeWorkflow:

    #: An instance of the MCO model information
    mco_model = Instance(NevergradMCOModel)

    def __init__(self):
        self.objective_function = GridValleyObjective()
        self.mco_model = NevergradMCOModel(factory=None)
        self.mco_model.parameters = self.objective_function.get_params()
        self.mco_model.kpis = self.objective_function.get_kpis()

    def evaluate(self, parameter_values):

        return self.objective_function.objective(parameter_values)
