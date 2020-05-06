from unittest import TestCase, mock

from traits.testing.unittest_tools import UnittestTools

from traitsui.api import View

from force_bdss.api import (
    CategoricalMCOParameterFactory,
    CategoricalMCOParameter,
)

from force_nevergrad.nevergrad_plugin import NevergradPlugin
from force_nevergrad.mco.ng_mco import NevergradMCO
from force_nevergrad.mco.ng_mco_factory import NevergradMCOFactory
from force_nevergrad.mco.ng_mco_model import NevergradMCOModel

from force_nevergrad.tests.probe_classes.workflow import ProbeWorkflow


class TestMCO(TestCase, UnittestTools):
    def setUp(self):
        self.plugin = NevergradPlugin()
        self.factory = self.plugin.mco_factories[0]
        self.mco = self.factory.create_optimizer()
        self.model = self.factory.create_model()

        self.parameters = [
            CategoricalMCOParameter(
                mock.Mock(spec=CategoricalMCOParameterFactory),
                categories=["A", "B"],
            )
        ]
        self.model.parameters = self.parameters

    def test_mco_model(self):
        self.assertEqual(100, self.model.budget)
        self.assertEqual(True, self.model.verbose_run)
        view = self.model.default_traits_view()
        self.assertIsInstance(view, View)

    def test_mco_factory(self):
        self.assertIsInstance(self.factory, NevergradMCOFactory)
        self.assertEqual("nevergrad_mco", self.factory.get_identifier())
        self.assertIs(self.factory.get_model_class(), NevergradMCOModel)
        self.assertIs(self.factory.get_optimizer_class(), NevergradMCO)
        self.assertEqual(5, len(self.factory.get_parameter_factory_classes()))

    def test_simple_run(self):

        # workflow and MCO to run it
        workflow = ProbeWorkflow()
        mco = self.factory.create_optimizer()

        # run the workflow and make sure model.notify_progress_event was fired
        # (as this is returning the pareto-set and we don't know how many
        # points are in the set, we don't know how many times it will be fired,
        # so we leave out the count arg in assertTraitChanges)
        with self.assertTraitChanges(workflow.mco_model, "event"):
            mco.run(workflow)
