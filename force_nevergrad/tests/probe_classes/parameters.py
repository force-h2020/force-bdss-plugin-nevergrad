
from force_bdss.mco.parameters.mco_parameters import BaseMCOParameter

from traits.api import List, Array, Float


class MyUnOrderedSetMCOParameter(BaseMCOParameter):
    set = List()


class MyOrderedSetMCOParameter(BaseMCOParameter):
    levels = List()


class MyScalarMCOParameter(BaseMCOParameter):
    x0 = Float()


class MyVectorMCOParameter(BaseMCOParameter):
    x0 = Array()


class MyArrayMCOParameter(BaseMCOParameter):
    value = Array()
