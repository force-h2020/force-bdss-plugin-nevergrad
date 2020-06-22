#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from unittest import TestCase
import numpy as np

from force_nevergrad.engine.nevergrad_optimizers import (
    NevergradMultiOptimizer,
    NevergradScalarOptimizer,
    translate_mco_to_ng,
    translate_ng_to_mco
)

from force_nevergrad.tests.probe_classes.optimizer import (
    TwoMinimaObjective,
    GridValleyObjective
)

from force_bdss.mco.parameters.mco_parameters import (
    FixedMCOParameter,
    RangedMCOParameter,
    RangedVectorMCOParameter,
    ListedMCOParameter,
    CategoricalMCOParameter
)

from force_nevergrad.tests.probe_classes.parameters import (
    MyUnOrderedSetMCOParameter,
    MyOrderedSetMCOParameter,
    MyScalarMCOParameter,
    MyVectorMCOParameter,
    MyArrayMCOParameter,
)


def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


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

    def test_parametrization(self):

        # 11 parameters, with no. of values = ...
        params = [
            FixedMCOParameter(
                factory=None,
                value=1.0
            ),                          # ... 1
            RangedMCOParameter(
                factory=None,
                initial_value=1.0
            ),                          # ... 1
            RangedVectorMCOParameter(
                factory=None,
                initial_value=[1.0 for i in range(10)]
            ),                          # ... 10
            ListedMCOParameter(
                factory=None,
                levels=[i for i in range(10)]
            ),                          # ... 1
            CategoricalMCOParameter(
                factory=None,
                categories=['a', 'b', 'c', 'd']
            ),                          # ... 1
            MyUnOrderedSetMCOParameter(
                factory=None,
                set=['no', 'good', 'foo']
            ),                          # ... 1
            MyOrderedSetMCOParameter(
                factory=None,
                levels=['a', 'b', 'c']
            ),                          # ... 1
            MyScalarMCOParameter(
                factory=None,
                x0=0.0
            ),                          # ... 1
            MyVectorMCOParameter(
                factory=None,
                x0=np.zeros((5, ))
            ),                          # ... 5
            MyArrayMCOParameter(
                factory=None,
                value=np.zeros((3, 3))
            ),                          # ... 9
            86              # counts as "some other object"
            #                           # ... 1
        ]                               # sum = 32

        # translate to nevergrad
        instrumentation = translate_mco_to_ng(params)

        # translate back values
        mco_values = translate_ng_to_mco(instrumentation.args)

        # is the number of parameters correct?
        self.assertEqual(11, len(mco_values))

        # is the total number of parameter values correct?
        self.assertEqual(32, len(flatten(mco_values)))

        # is the listed parameter value (index 3)
        # less than those allowed?
        self.assertLess(mco_values[3], 10)

        # is the non-standard, 3x3 array parameter (index 9)
        # converted to a [[],[],[]]?
        self.assertEqual(3, len(mco_values[9]))
        self.assertEqual(3, len(mco_values[9][0]))

        # is the non-recogisable parameter set to null constant?
        self.assertEqual(mco_values[10], 'null')

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
            self.assertEqual(1, len(optimal))
            optimum = optimal[0]

            # the position of the actual global minimum of the objective
            # and the tolerance (allowable distance from)
            global_optimum, tolerance = foo.get_global_optimum()

            # is this the global optimum?
            self.assertEqual(len(global_optimum), len(optimum))
            # ...compare parameter values
            for parameter in zip(optimum, global_optimum):
                # parameter is a list (RangedVector, Listed, Categorical)
                if isinstance(parameter[0], list):
                    self.assertEqual(len(parameter[1]), len(parameter[0]))
                    for i in range(len(parameter[0])):
                        self.assertAlmostEqual(parameter[1][i],
                                               parameter[0][i],
                                               delta=tolerance)
                # parameter is a scalar
                # cannot test grid-valley objective, because the
                # parameters are ListedMCOParameter; there is no
                # way to set the initial index of such a parameter;
                # and therefore no way to control which optimum
                # (global or local) is reached.
                else:
                    if foo.__class__.__name__ == 'GridValleyObjective':
                        continue
                    self.assertAlmostEqual(parameter[1], parameter[0],
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
