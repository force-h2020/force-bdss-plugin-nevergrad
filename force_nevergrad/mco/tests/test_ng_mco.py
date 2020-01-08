from unittest import TestCase, mock

from traits.testing.unittest_tools import UnittestTools

from force_bdss.api import (
    WorkflowEvaluator,
    KPISpecification,
    Workflow,
    DataValue,
    FixedMCOParameterFactory,
    ListedMCOParameterFactory,
    RangedMCOParameterFactory,
    FixedMCOParameter,
    ListedMCOParameter,
    RangedMCOParameter,
    CategoricalMCOParameterFactory,
    CategoricalMCOParameter
)

from force_nevergrad.nevergrad_plugin import NevergradPlugin
from force_nevergrad.mco.ng_mco import NevergradMCO
from force_nevergrad.mco.ng_mco_factory import NevergradMCOFactory
from force_nevergrad.mco.ng_mco_model import NevergradMCOModel


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
            ),
            FixedMCOParameter(
                mock.Mock(spec=FixedMCOParameterFactory), value=12.0
            ),
            ListedMCOParameter(
                mock.Mock(spec=ListedMCOParameterFactory), levels=[0.1, 2.5]
            ),
            RangedMCOParameter(
                mock.Mock(spec=RangedMCOParameterFactory),
                upper_bound=1.5,
                n_samples=3,
            ),
        ]
        self.model.parameters = self.parameters

    def test_mco_model(self):
        self.assertEqual("TwoPointsDE", self.model.algorithms)
        self.assertEqual(100, self.model.budget)
        self.assertEqual(True, self.model.verbose_run)

    def test_mco_factory(self):
        self.assertIsInstance(self.factory, NevergradMCOFactory)
        self.assertEqual("nevergrad_mco", self.factory.get_identifier())
        self.assertIs(self.factory.get_model_class(), NevergradMCOModel)
        self.assertIs(self.factory.get_optimizer_class(), NevergradMCO)
        self.assertEqual(4, len(self.factory.get_parameter_factory_classes()))

    def test_simple_run(self):
        mco = self.factory.create_optimizer()
        model = self.factory.create_model()
        model.budget = 61
        model.parameters = self.parameters
        model.kpis = [KPISpecification(), KPISpecification()]

        evaluator = WorkflowEvaluator(
            workflow=Workflow(), workflow_filepath="whatever"
        )
        evaluator.workflow.mco = model
        kpis = [DataValue(value=1), DataValue(value=2)]
        with self.assertTraitChanges(mco, "event", count=61):
            with mock.patch(
                "force_bdss.api.Workflow.execute", return_value=kpis
            ) as mock_exec:
                mco.run(evaluator)
                self.assertEqual(61, mock_exec.call_count)
