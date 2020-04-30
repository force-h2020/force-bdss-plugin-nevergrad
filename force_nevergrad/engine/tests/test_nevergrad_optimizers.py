from unittest import TestCase, mock

import nevergrad as ng
from nevergrad.parametrization.transforms import ArctanBound
from nevergrad.parametrization import core as ng_core

from force_bdss.api import (
    KPISpecification,
    FixedMCOParameterFactory,
    RangedMCOParameterFactory,
    ListedMCOParameterFactory,
    CategoricalMCOParameterFactory,
    RangedVectorMCOParameterFactory,
)
from force_bdss.tests.dummy_classes.mco import DummyMCOFactory
from force_bdss.tests.dummy_classes.optimizer_engine import (
    MixinDummyOptimizerEngine,
)

from force_nevergrad.engine.nevergrad_optimizers import (
    NevergradTypeError,
    create_instrumentation_variable
)

from force_bdss.mco.optimizer_engines.aposteriori_optimizer_engine import (
    AposterioriOptimizerEngine
)
from force_nevergrad.engine.nevergrad_optimizers import (
    NevergradMultiOptimizer
)


class DummyOptimizerEngine(
    MixinDummyOptimizerEngine, AposterioriOptimizerEngine
):
    pass


class TestNevergradOptimizerEngine(TestCase):
    def setUp(self):
        self.plugin = {"id": "pid", "name": "Plugin"}
        self.factory = DummyMCOFactory(self.plugin)

        self.kpis = [KPISpecification(), KPISpecification()]
        self.parameters = [1, 1, 1, 1]

        self.parameters = [
            RangedMCOParameterFactory(self.factory).create_model(
                {"lower_bound": 0.0, "upper_bound": 1.0}
            )
            for _ in self.parameters
        ]

        optim = NevergradMultiOptimizer(
            kpis=self.kpis,
        )

        self.mocked_engine = DummyOptimizerEngine(
            parameters=self.parameters, kpis=self.kpis,
            optimizer=optim
        )

    def test_init(self):
        self.assertEqual("TwoPointsDE", self.engine.optimizer.algorithms)
        self.assertEqual(500, self.engine.optimizer.budget)

    def test_optimize(self):
        self.mocked_engine.verbose_run = True
        optimized_data = list(self.mocked_engine.optimize())

        self.mocked_engine.verbose_run = False
        for optimized_data in self.mocked_engine.optimize():
            self.assertEqual(4, len(optimized_data[0]))
            self.assertEqual(2, len(optimized_data[1]))

    def test_optimize_vector(self):
        vector_parameter = RangedVectorMCOParameterFactory(
            self.factory
        ).create_model(
            {
                "lower_bound": [0.0 for _ in self.parameters],
                "upper_bound": [1.0 for _ in self.parameters],
            }
        )
        self.engine.parameters = [vector_parameter]
        self.mocked_engine.verbose_run = True
        optimized_data = list(self.mocked_engine.optimize())
        self.mocked_engine.verbose_run = False
        for optimized_data in self.mocked_engine.optimize():
            self.assertEqual(4, len(optimized_data[0]))
            self.assertEqual(2, len(optimized_data[1]))

    def test__create_instrumentation_variable(self):
        mock_factory = mock.Mock(
            spec=self.factory,
            plugin_id="pid",
            plugin_name="Plugin",
            id="mcoid",
        )
        fixed_variable = FixedMCOParameterFactory(mock_factory).create_model(
            data_values={"value": 42}
        )
        fixed_variable = create_instrumentation_variable(
            fixed_variable
        )
        self.assertIsInstance(fixed_variable, ng_core.Constant)
        self.assertEqual(42, fixed_variable.value)

        ranged_variable = RangedMCOParameterFactory(mock_factory).create_model(
            data_values={"lower_bound": -1.0, "upper_bound": 3.14}
        )
        ranged_variable = create_instrumentation_variable(
            ranged_variable
        )
        self.assertIsInstance(ranged_variable, ng.p.Scalar)
        self.assertListEqual(
            [3.14], list(ranged_variable.bound_transform.a_max)
        )
        self.assertListEqual(
            [-1.0], list(ranged_variable.bound_transform.a_min)
        )
        self.assertIsInstance(ranged_variable.bound_transform, ArctanBound)

        listed_variable = ListedMCOParameterFactory(mock_factory).create_model(
            data_values={"levels": [2.0, 1.0, 0.0]}
        )
        listed_variable = create_instrumentation_variable(
            listed_variable
        )
        self.assertIsInstance(listed_variable, ng.p.TransitionChoice)
        self.assertListEqual(
            [0.0, 1.0, 2.0], list(listed_variable.choices.value)
        )

        categorical_variable = CategoricalMCOParameterFactory(
            mock_factory
        ).create_model(data_values={"categories": ["2.0", "1.0", "0.0"]})
        categorical_variable = create_instrumentation_variable(
            categorical_variable
        )
        self.assertIsInstance(categorical_variable, ng.p.Choice)
        self.assertListEqual(
            ["2.0", "1.0", "0.0"], list(categorical_variable.choices.value)
        )

        lower_bound = [0.0 for _ in self.parameters]
        upper_bound = [1.0 for _ in self.parameters]
        vector_variable = RangedVectorMCOParameterFactory(
            self.factory
        ).create_model(
            {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )
        vector_variable = create_instrumentation_variable(
            vector_variable
        )
        self.assertIsInstance(vector_variable, ng.p.Array)
        self.assertFalse(isinstance(vector_variable, ng.p.Scalar))
        self.assertListEqual(
            upper_bound, list(vector_variable.bound_transform.a_max[0])
        )
        self.assertListEqual(
            lower_bound, list(vector_variable.bound_transform.a_min[0])
        )
        self.assertIsInstance(vector_variable.bound_transform, ArctanBound)

        with self.assertRaises(NevergradTypeError):
            create_instrumentation_variable(1)
