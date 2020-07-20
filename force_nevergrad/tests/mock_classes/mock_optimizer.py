from collections import namedtuple

from force_nevergrad.engine.parameter_translation import (
    translate_mco_to_ng,
)

MockPoint = namedtuple('x', ['args', 'kwargs'])


class MockOptimizer:

    def __init__(self, params):
        self.ng_params = translate_mco_to_ng(params)

    def minimize(self, *vargs):
        return self.ng_params

    def ask(self):
        return MockPoint(
            [x for x in range(len(self.ng_params))],
            {}
        )

    def tell(self, x, volume):
        return


class MockMultiObjectiveFunction:

    def __init__(self, params, pareto_size=10):
        self.ng_params = translate_mco_to_ng(params)
        self.pareto_size = pareto_size

    def multiobjective_function(self, *args):
        return 1

    def compute_aggregate_loss(self, *args, **kwargs):
        return 1

    def pareto_front(self):
        return [(self.ng_params.args, self.ng_params.kwargs)]*self.pareto_size
