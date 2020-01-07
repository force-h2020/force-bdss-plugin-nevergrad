from force_bdss.api import (
    BaseMCOFactory,
    FixedMCOParameterFactory,
    ListedMCOParameterFactory,
    RangedMCOParameterFactory,
    CategoricalMCOParameterFactory,
    BaseMCOCommunicator
)

from .ng_mco_model import NevergradMCOModel
from .ng_mco import NevergradMCO


class NevergradMCOFactory(BaseMCOFactory):
    def get_identifier(self):
        return "nevergrad_mco"

    def get_name(self):
        return "Gradient Free Multi Criteria optimizer"

    #: Returns the model class
    def get_model_class(self):
        return NevergradMCOModel

    #: Returns the optimizer class
    def get_optimizer_class(self):
        return NevergradMCO

    #: Returns the communicator class
    def get_communicator_class(self):
        return BaseMCOCommunicator

    #: Factory classes of the parameters the MCO supports.
    def get_parameter_factory_classes(self):
        return [
            FixedMCOParameterFactory,
            ListedMCOParameterFactory,
            RangedMCOParameterFactory,
            CategoricalMCOParameterFactory,
        ]
