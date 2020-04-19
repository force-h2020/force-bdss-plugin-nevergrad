from force_bdss.api import BaseExtensionPlugin, plugin_id

from force_nevergrad.mco.ng_mco_factory import NevergradMCOFactory
from force_nevergrad.gauss2d.gauss2d_data_source_factory import Gauss2dFactory

PLUGIN_VERSION = 0


class NevergradPlugin(BaseExtensionPlugin):
    """This plugin provides useful classes and DataSource subclasses
    for using the Nevergrad library.
    """

    id = plugin_id("nevergrad", "wrapper", PLUGIN_VERSION)

    def get_name(self):
        return "Nevergrad Plugin"

    def get_description(self):
        return (
            "A plugin containing force-bdss compatible objects "
            "using functionalities of the Nevergrad library"
        )

    def get_version(self):
        return PLUGIN_VERSION

    #: Define the factory classes that you want to export to this list.
    def get_factory_classes(self):
        return [NevergradMCOFactory,
                Gauss2dFactory]
