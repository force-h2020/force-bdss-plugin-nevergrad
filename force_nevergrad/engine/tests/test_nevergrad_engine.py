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
)
from force_bdss.tests.dummy_classes.mco import DummyMCOFactory
from force_bdss.tests.dummy_classes.optimizer_engine import (
    MixinDummyOptimizerEngine,
)
from force_nevergrad.engine.nevergrad_engine import (
    NevergradTypeError,
    NevergradOptimizerEngine,
)


class DummyOptimizerEngine(
    MixinDummyOptimizerEngine, NevergradOptimizerEngine
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

        self.optimizer = NevergradOptimizerEngine(
            parameters=self.parameters, kpis=self.kpis
        )
        self.mocked_optimizer = DummyOptimizerEngine(
            parameters=self.parameters, kpis=self.kpis
        )

    def test_init(self):
        self.assertIsInstance(self.optimizer, NevergradOptimizerEngine)
        self.assertEqual("Nevergrad", self.optimizer.name)
        self.assertIs(self.optimizer.single_point_evaluator, None)
        self.assertEqual("TwoPointsDE", self.optimizer.algorithms)
        self.assertEqual(500, self.optimizer.budget)

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
        fixed_variable = self.optimizer._create_instrumentation_variable(
            fixed_variable
        )
        self.assertIsInstance(fixed_variable, ng_core.Constant)
        self.assertEqual(42, fixed_variable.value)

        ranged_variable = RangedMCOParameterFactory(mock_factory).create_model(
            data_values={"lower_bound": -1.0, "upper_bound": 3.14}
        )
        ranged_variable = self.optimizer._create_instrumentation_variable(
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
        listed_variable = self.optimizer._create_instrumentation_variable(
            listed_variable
        )
        self.assertIsInstance(listed_variable, ng.p.TransitionChoice)
        self.assertListEqual(
            [0.0, 1.0, 2.0], list(listed_variable.choices.value)
        )

        categorical_variable = CategoricalMCOParameterFactory(
            mock_factory
        ).create_model(data_values={"categories": ["2.0", "1.0", "0.0"]})
        categorical_variable = self.optimizer._create_instrumentation_variable(
            categorical_variable
        )
        self.assertIsInstance(categorical_variable, ng.p.Choice)
        self.assertListEqual(
            ["2.0", "1.0", "0.0"], list(categorical_variable.choices.value)
        )

        with self.assertRaises(NevergradTypeError):
            self.optimizer._create_instrumentation_variable(1)

    def test__create_instrumentation(self):
        instrumentation = self.optimizer._assemble_instrumentation()
        self.assertIsInstance(instrumentation, ng.p.Instrumentation)
        self.assertEqual(
            len(self.optimizer.parameters), len(instrumentation.args)
        )
        for i, parameter in enumerate(self.optimizer.parameters):
            self.assertListEqual(
                [parameter.upper_bound],
                list(instrumentation[0][i].bound_transform.a_max),
            )
            self.assertListEqual(
                [parameter.lower_bound],
                list(instrumentation[0][i].bound_transform.a_min),
            )

        # Create instrumentation from unbound parameters
        parameter = self.optimizer.parameters[0]
        instrumentation = self.optimizer._assemble_instrumentation([parameter])
        args = instrumentation[0]
        self.assertEqual(1, len(args))
        self.assertListEqual(
            [parameter.upper_bound], list(args[0].bound_transform.a_max)
        )

    def test__create_kpi_bounds(self):
        self.optimizer.kpis[0].scale_factor = 10
        bounds = self.optimizer.kpi_bounds
        self.assertEqual(len(self.optimizer.kpis), len(bounds))
        for kpi, kpi_bound in zip(self.optimizer.kpis, bounds):
            self.assertEqual(kpi.scale_factor, kpi_bound)

    def test_optimize(self):
        self.mocked_optimizer.verbose_run = True
        optimized_data = list(self.mocked_optimizer.optimize())
        self.assertEqual(self.mocked_optimizer.budget, len(optimized_data))

        self.mocked_optimizer.verbose_run = False
        for optimized_data in self.mocked_optimizer.optimize():
            self.assertEqual(4, len(optimized_data[0]))
            self.assertEqual(2, len(optimized_data[1]))
