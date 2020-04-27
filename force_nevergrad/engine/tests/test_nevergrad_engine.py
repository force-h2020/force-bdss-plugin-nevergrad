from unittest import TestCase

from force_bdss.api import (
    KPISpecification,
    RangedMCOParameterFactory,
    RangedVectorMCOParameterFactory,
)
from force_bdss.tests.dummy_classes.mco import DummyMCOFactory
from force_bdss.tests.dummy_classes.optimizer_engine import (
    MixinDummyOptimizerEngine,
)

from force_nevergrad.engine.aposteriori_nevergrad_engine import (
    AposterioriNevergradEngine
)


class DummyOptimizerEngine(
    MixinDummyOptimizerEngine, AposterioriNevergradEngine
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

        self.optimizer = AposterioriNevergradEngine(
            parameters=self.parameters, kpis=self.kpis
        )
        self.mocked_optimizer = DummyOptimizerEngine(
            parameters=self.parameters, kpis=self.kpis
        )

    def test_init(self):
        self.assertIsInstance(self.optimizer, AposterioriNevergradEngine)
        self.assertEqual("TwoPointsDE", self.optimizer.algorithms)
        self.assertEqual(500, self.optimizer.budget)

    def test_optimize(self):
        self.mocked_optimizer.verbose_run = True
        optimized_data = list(self.mocked_optimizer.optimize())

        self.mocked_optimizer.verbose_run = False
        for optimized_data in self.mocked_optimizer.optimize():
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
        self.optimizer.parameters = [vector_parameter]
        self.mocked_optimizer.verbose_run = True
        optimized_data = list(self.mocked_optimizer.optimize())
        self.mocked_optimizer.verbose_run = False
        for optimized_data in self.mocked_optimizer.optimize():
            self.assertEqual(4, len(optimized_data[0]))
            self.assertEqual(2, len(optimized_data[1]))
