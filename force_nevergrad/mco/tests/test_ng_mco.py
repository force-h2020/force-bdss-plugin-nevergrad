from unittest import TestCase, mock

from traits.testing.unittest_tools import UnittestTools

from force_bdss.api import (
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
    CategoricalMCOParameter,
    RangedVectorMCOParameter,
    RangedVectorMCOParameterFactory
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
            RangedVectorMCOParameter(
                mock.Mock(spec=RangedVectorMCOParameterFactory),
                upper_bound=[1.5],
                n_samples=3,
            )
        ]
        self.model.parameters = self.parameters

    def test_mco_model(self):
        self.assertEqual(100, self.model.budget)
        self.assertEqual(True, self.model.verbose_run)

    def test_mco_factory(self):
        self.assertIsInstance(self.factory, NevergradMCOFactory)
        self.assertEqual("nevergrad_mco", self.factory.get_identifier())
        self.assertIs(self.factory.get_model_class(), NevergradMCOModel)
        self.assertIs(self.factory.get_optimizer_class(), NevergradMCO)
        self.assertEqual(5, len(self.factory.get_parameter_factory_classes()))

    def test_simple_run(self):
        mco = self.factory.create_optimizer()
        model = self.factory.create_model()
        model.budget = 61
        model.parameters = self.parameters
        model.kpis = [KPISpecification(), KPISpecification()]

        workflow = Workflow()
        workflow.mco_model = model
        kpis = [DataValue(value=1), DataValue(value=2)]
        with self.assertTraitChanges(model, "event", count=0):
            with mock.patch(
                    "force_bdss.api.Workflow.execute", return_value=kpis
            ) as mock_exec:
                mco.run(workflow)
                self.assertEqual(61, mock_exec.call_count)
