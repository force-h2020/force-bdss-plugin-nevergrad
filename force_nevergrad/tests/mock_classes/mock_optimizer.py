from force_nevergrad.engine.parameter_translation import (
    translate_mco_to_ng,
)


class MockOptimizer:

    def __init__(self, params):
        self.ng_params = translate_mco_to_ng(params)

    def minimize(self, *vargs):

        return self.ng_params


class MockMultiObjectiveFunction:

    def __init__(self, params, pareto_size=10):
        self.ng_params = translate_mco_to_ng(params)
        self.pareto_size = pareto_size

    def pareto_front(self):

        return [(self.ng_params.args, self.ng_params.kwargs)]*self.pareto_size
