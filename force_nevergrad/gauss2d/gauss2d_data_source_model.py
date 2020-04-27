from traits.api import Float
from traitsui.api import View, Item

from force_bdss.api import BaseDataSourceModel


class Gauss2dDataSourceModel(BaseDataSourceModel):

    g1_peak = Float(-2.0)
    g1_cent_x = Float(-1.0)
    g1_cent_y = Float(-1.0)
    g1_sigm_x = Float(0.6)
    g1_sigm_y = Float(0.6)

    g2_peak = Float(-1.0)
    g2_cent_x = Float(1.0)
    g2_cent_y = Float(1.0)
    g2_sigm_x = Float(1.2)
    g2_sigm_y = Float(1.2)

    traits_view = View(
        Item("g1_peak"),
        Item("g1_cent_x"),
        Item("g1_cent_y"),
        Item("g1_sigm_x"),
        Item("g1_sigm_y"),
        Item("g2_peak"),
        Item("g2_cent_x"),
        Item("g2_cent_y"),
        Item("g2_sigm_x"),
        Item("g2_sigm_y"),
    )
