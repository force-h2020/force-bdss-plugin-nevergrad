import math

from force_bdss.api import DataValue, Slot, BaseDataSource


class Gauss2dDataSource(BaseDataSource):

    def run(self, model, parameters):
        x = parameters[0].value
        y = parameters[1].value

        a1 = ((x - model.g1_cent_x)**2)/(2.0*model.g1_sigm_x**2)
        a1 += ((y - model.g1_cent_y)**2) / (2.0*model.g1_sigm_y**2)
        a1 = model.g1_peak * math.exp(-a1)

        a2 = ((x - model.g2_cent_x)**2) / (2.0*model.g2_sigm_x**2)
        a2 += ((y - model.g2_cent_y)**2) / (2.0*model.g2_sigm_y**2)
        a2 = model.g2_peak * math.exp(-a2)

        return [
            DataValue(value=a1, type="CONCENTRATION"),
            DataValue(value=a2, type="CONCENTRATION")
        ]

    def slots(self, model):
        return (
            (
                Slot(description="x", type="COORDINATE"),
                Slot(description="y", type="COORDINATE"),
            ),
            (
                Slot(description="a1", type="CONCENTRATION"),
                Slot(description="a2", type="CONCENTRATION"),
            )
        )
