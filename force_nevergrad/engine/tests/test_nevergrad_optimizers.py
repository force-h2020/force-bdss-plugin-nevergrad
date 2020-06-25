#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from unittest import TestCase
from unittest.mock import Mock, patch
import numpy as np
from functools import partial

from force_nevergrad.engine.nevergrad_optimizers import (
    nevergrad_function,
    NevergradMultiOptimizer,
    NevergradScalarOptimizer,
)

from force_nevergrad.engine.parameter_translation import (
    translate_mco_to_ng,
)

from force_nevergrad.tests.mock_classes.mock_optimizer import (
    MockOptimizer,
    MockMultiObjectiveFunction
)

from nevergrad.optimization.base import Optimizer
from nevergrad.functions import MultiobjectiveFunction


class TestNevergradOptimizer(TestCase):

    def setUp(self):

        # mock 'MCO' parameters and instrumentation
        self.params = [
            Mock(**{'x0': 0.0}),
            Mock(**{'value': np.zeros((3, 3))}),
        ]
        self.instrumentation = translate_mco_to_ng(self.params)

        # stub function
        self.m_foo = Mock(**{'return_value': [1, 2, 3]})

    def test_nevergrad_function(self):

        # scalar (summed) objective
        objective = nevergrad_function(
            *[],
            function=self.m_foo,
            is_scalar=True,
            minimize_objectives=[True, True, True]
        )
        self.assertEqual(objective, 6)

        # maximize the 2nd objective
        objective = nevergrad_function(
            *[],
            function=self.m_foo,
            is_scalar=True,
            minimize_objectives=[True, False, True]
        )
        self.assertEqual(objective, 2)

        # multi-objective
        objective = nevergrad_function(
            *[],
            function=self.m_foo,
            is_scalar=False,
            minimize_objectives=[True, False, True]
        )
        self.assertListEqual(objective, [1, -2, 3])

    @patch.object(
        NevergradScalarOptimizer,
        'get_optimizer',
        return_value=MockOptimizer(
            params=[
                Mock(**{'x0': 0.0}),
                Mock(**{'value': np.zeros((3, 3))}),
            ]
        )
    )
    def test_nevergrad_scalar_optimizer(self, mock_optimizer):

        # IOptimizer that optimizes with MockOptimizer.minimize()
        optimizer = NevergradScalarOptimizer()

        # default algorithm
        self.assertEqual(optimizer._algorithms_default(), "TwoPointsDE")

        # optimize
        count = 0
        for x in optimizer.optimize_function(self.m_foo, [1.0]):
            # x0 of first parameter
            self.assertEqual(x[0], 0.0)
            # value of second parameter
            self.assertListEqual(
                x[1],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
            count += 1

        # only one optimal point should be returned
        self.assertEqual(count, 1)

    @patch.object(
        NevergradMultiOptimizer,
        'get_optimizer',
        return_value=MockOptimizer(
            params=[
                Mock(**{'x0': 0.0}),
                Mock(**{'value': np.zeros((3, 3))}),
            ]
        )
    )
    @patch.object(
        NevergradMultiOptimizer,
        'get_multiobjective_function',
        return_value=MockMultiObjectiveFunction(
            params=[
                Mock(**{'x0': 0.0}),
                Mock(**{'value': np.zeros((3, 3))}),
            ],
            pareto_size=10,
        )
    )
    def test_nevergrad_multi_optimizer(self, mock1, mock2):

        # IOptimizer that optimizes with MockOptimizer.minimize()
        # and returns a pareto front from MockMockMultiObjectiveFunction
        optimizer = NevergradMultiOptimizer()

        # default algorithm
        self.assertEqual(optimizer._algorithms_default(), "TwoPointsDE")

        # optimize
        count = 0
        for x in optimizer.optimize_function(self.m_foo, [1.0]):
            # x0 of first parameter
            self.assertEqual(x[0], 0.0)
            # value of second parameter
            self.assertListEqual(
                x[1],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
            count += 1

        # ten points in Pareto front
        self.assertEqual(count, 10)

    def test_get_optimizer(self):

        optimizer = NevergradScalarOptimizer()
        ng_optimizer = optimizer.get_optimizer(self.params)
        self.assertIsInstance(ng_optimizer, Optimizer)
        self.assertEqual(ng_optimizer.dimension, 10)

        optimizer = NevergradMultiOptimizer()
        ng_optimizer = optimizer.get_optimizer(self.params)
        self.assertIsInstance(ng_optimizer, Optimizer)
        self.assertEqual(ng_optimizer.dimension, 10)

    def test_get_multiobjective_function(self):

        # optimizer
        optimizer = NevergradMultiOptimizer()

        # multi-objective
        ng_func = partial(
            nevergrad_function,
            function=self.m_foo,
            is_scalar=False,
            minimize_objectives=[True, False, True]
        )

        # get multi-objective function object
        multi_objective = optimizer.get_multiobjective_function(ng_func)
        self.assertIsInstance(multi_objective, MultiobjectiveFunction)
