from unittest import TestCase

from force_nevergrad.engine.nevergrad_optimizers import (
    NevergradMultiOptimizer,
    NevergradScalarOptimizer,

)

from force_nevergrad.tests.probe_classes.optimizer import (
    TwoMinimaObjective,
    GridValleyObjective
)


class TestNevergradOptimizer(TestCase):
    def setUp(self):

        # some objective functions to test
        self.functions_to_test = [
            TwoMinimaObjective(),
            GridValleyObjective()
        ]

        # scalar- and multi-objective function optimizers
        self.scalar_optimizer = NevergradScalarOptimizer()
        self.multi_optimizer = NevergradMultiOptimizer()

    def test_init(self):
        self.assertEqual("TwoPointsDE", self.multi_optimizer.algorithms)
        self.assertEqual(500, self.multi_optimizer.budget)

    def test_scalar_objective(self):

        # test each objective function in turn...
        for foo in self.functions_to_test:

            # set kpis
            self.scalar_optimizer.kpis = foo.get_kpis()

            # get optimal point
            optimal = [
                p for p in self.scalar_optimizer.optimize_function(
                    foo.objective, foo.get_params()
                )
            ]

            # there should only be one point
            self.assertEqual(len(optimal), 1)
            optimum = optimal[0]

            # the position of the actual global minimum of the objective
            # and the tolerance (allowable distance from)
            global_optimum, tolerance = foo.get_global_optimum()

            # is this the global optimum?
            self.assertEqual(len(optimum), len(global_optimum))
            # ...compare parameter values
            for parameter in zip(optimum, global_optimum):
                # parameter is a list (RangedVector, Listed, Categorical)
                if isinstance(parameter[0], list):
                    self.assertEqual(len(parameter[0]), len(parameter[1]))
                    for i in range(len(parameter[0])):
                        self.assertAlmostEqual(parameter[0][i],
                                               parameter[1][i],
                                               delta=tolerance)
                # parameter is a scalar
                else:
                    self.assertAlmostEqual(parameter[0], parameter[1],
                                           delta=tolerance)

    def test_multi_objective(self):

        # test each objective function in turn...
        for foo in self.functions_to_test:

            # set kpis
            self.multi_optimizer.kpis = foo.get_kpis()

            # get Pareto set (of points in parameter space)
            pareto = [
                p for p in self.multi_optimizer.optimize_function(
                        foo.objective, foo.get_params()
                )
            ]

            # there should be more than one point in the Pareto-set
            self.assertGreater(len(pareto), 1)

            # are all the points in the Pareto set?
            for p in pareto:
                self.assertTrue(foo.is_pareto_optimal(p))
