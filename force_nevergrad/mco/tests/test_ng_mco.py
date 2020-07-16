#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from unittest import TestCase, mock
from io import StringIO
from unittest.mock import patch

from traits.testing.unittest_tools import UnittestTools

from traitsui.api import View

from force_bdss.api import (
    CategoricalMCOParameterFactory,
    DataValue,
    FixedMCOParameter,
    RangedMCOParameter,
    RangedVectorMCOParameter,
    ListedMCOParameter,
    CategoricalMCOParameter,
    BaseMCOParameter
)

from force_nevergrad.nevergrad_plugin import NevergradPlugin
from force_nevergrad.mco.ng_mco import NevergradMCO, NevergradOptimizerEngine
from force_nevergrad.mco.ng_mco_factory import NevergradMCOFactory
from force_nevergrad.mco.ng_mco_model import NevergradMCOModel
from force_nevergrad.mco.ng_mco_communicator import NevergradMCOCommunicator

from force_nevergrad.tests.probe_classes.workflow import ProbeWorkflow


class TestNevergradOptimizerEngine(TestCase):
    def setUp(self):
        self.model = ProbeWorkflow()

    def test_score_upper_bounds(self):
        kpis = self.model.mco_model.kpis
        engine = NevergradOptimizerEngine(
            kpis=kpis
        )

        self.assertListEqual(
            [None, None], engine.score_upper_bounds())

        kpis[0].use_bounds = True
        self.assertListEqual(
            [2.5, None], engine.score_upper_bounds())

        kpis[0].objective = "MAXIMISE"
        self.assertListEqual(
            [-0.5, None], engine.score_upper_bounds())

        kpis[0].objective = "TARGET"
        kpis[0].target_value = 1.5
        self.assertListEqual(
            [2.5, None], engine.score_upper_bounds())


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

        # Check upper bound assignments do not break workflow.
        # One upper bound assigned by user, the other will be estimated
        workflow.mco_model.kpis[0].use_bounds = True
        with self.assertTraitChanges(workflow.mco_model, "event"):
            mco.run(workflow)

        # Both upper bounds assigned by user
        workflow.mco_model.kpis[1].use_bounds = True
        with self.assertTraitChanges(workflow.mco_model, "event"):
            mco.run(workflow)

        # Handle MAXIMISE KPI objectives
        workflow.mco_model.kpis.objective = 'MAXIMISE'
        with self.assertTraitChanges(workflow.mco_model, "event"):
            mco.run(workflow)

    def test_communicator(self):

        # communicator
        comm = NevergradMCOCommunicator(self.factory)

        # receive_from_mco: get parameter values from stdin ....
        # ...five model parameters (of all flavors!)
        self.model.parameters = [
            FixedMCOParameter(value=0.0, factory=None),
            RangedMCOParameter(initial_value=0.0, factory=None),
            RangedVectorMCOParameter(initial_value=[0.0, 0.0], factory=None),
            ListedMCOParameter(levels=[0.0, 0.0], factory=None),
            CategoricalMCOParameter(categories=['a', 'b'], factory=None),
            BaseMCOParameter(factory=None)
        ]
        # ...only supply values for the first two parameters.
        with patch('sys.stdin', StringIO('-1.0,-1.0')):
            inputs = comm.receive_from_mco(self.model)
            # First two values should be set from stdin;
            # Last value (BaseMCOParameter should be 0.0
            # Remaining values should be set from parameter:
            # initial value (ranged, etc) or first element (listed/categorical)
            self.assertEqual(
                [-1.0, -1.0, [0.0, 0.0], 0.0, 'a', 0.0],
                [x.value for x in inputs]
            )

        # send_to_mco: get kpis from stdout ....
        # ...two KPIs
        kpis = [
            DataValue(value=1.0),
            DataValue(value=1.0),
        ]
        with patch('sys.stdout', new_callable=StringIO) as stdout:
            comm.send_to_mco(self.model, kpis)
            # return should be tab-delimited line of KPIs
            self.assertEqual('1.0\t1.0\n', stdout.getvalue())
