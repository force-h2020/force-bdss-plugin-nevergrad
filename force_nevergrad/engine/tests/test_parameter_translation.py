from unittest import TestCase

from unittest.mock import Mock

from types import MethodType
import numpy as np

from ..parameter_translation import (
    get_attribute,
    duck_type_param,
    translate_mco_to_ng,
    translate_ng_to_mco,
)

from nevergrad import p as ngp

from force_bdss.mco.parameters.mco_parameters import (
    FixedMCOParameter,
    RangedMCOParameter,
    RangedVectorMCOParameter,
    ListedMCOParameter,
    CategoricalMCOParameter
)


def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


class TestParameterTranslation(TestCase):

    def setUp(self):
        pass

    def test_get_attribute(self):

        duck = Mock(**{
            'legs': 2,
            'make_sounds': Mock(spec=MethodType, return_value='quack')
        })

        # duck-type attribute
        self.assertIs(
            get_attribute(duck, {'fins', 'legs', 'hands'}), 2
        )

        # duck-type method return
        self.assertIs(
            get_attribute(duck, {'fins', 'make_sounds', 'hands'}), 'quack'
        )

        # no match
        self.assertIsNone(get_attribute(duck, {'fins', 'hands'}))

    def test_duck_type_param(self):

        param = duck_type_param(Mock(**{'choices': ['a', 'b', 2, 3]}))
        self.assertIsInstance(param, ngp.Choice)
        self.assertTupleEqual(param.choices.value, ('a', 'b', 2, 3))

        param = duck_type_param(Mock(**{'levels': [0, 1, 2, 3]}))
        self.assertIsInstance(param, ngp.TransitionChoice)
        self.assertTupleEqual(param.choices.value, (0, 1, 2, 3))

        param = duck_type_param(Mock(**{'initial_value': 1.0}))
        self.assertIsInstance(param, ngp.Scalar)
        self.assertEqual(param.value, 1.0)

        param = duck_type_param(Mock(**{'value': [7, 8, 9]}))
        self.assertIsInstance(param, ngp.Array)
        self.assertEqual(param.value.size, 3)

        param = duck_type_param(Mock(**{'x0': np.zeros((2, 2))}))
        self.assertIsInstance(param, ngp.Array)
        self.assertEqual(param.value.size, 4)

    def test_translate(self):

        params = [
            FixedMCOParameter(
                factory=None,
                value=1.0
            ),  # ... 1
            RangedMCOParameter(
                factory=None,
                initial_value=1.0
            ),  # ... 1
            RangedVectorMCOParameter(
                factory=None,
                initial_value=[1.0 for i in range(10)]
            ),  # ... 10
            ListedMCOParameter(
                factory=None,
                levels=[i for i in range(10)]
            ),  # ... 1
            CategoricalMCOParameter(
                factory=None,
                categories=['a', 'b', 'c', 'd']
            ),  # ... 1
            Mock(**{'set': ['no', 'good', 'foo']}),
            # ... 1
            Mock(**{'levels': ['a', 'b', 'c']}),
            # ... 1
            Mock(**{'x0': 0.0}),
            # ... 1
            Mock(**{'x0': np.zeros((5,))}),
            # ... 5
            Mock(**{'value': np.zeros((3, 3))}),
            # ... 9
            86  # counts as "some other object"
            #                           # ... 1
        ]

        # translate to nevergrad instrumentation
        instrumentation = translate_mco_to_ng(params)
        self.assertIsInstance(instrumentation, ngp.Instrumentation)

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

        # is the non-recognisable parameter set to null constant?
        self.assertEqual(mco_values[10], 'null')
