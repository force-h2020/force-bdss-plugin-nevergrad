#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

from force_bdss.api import (
    BaseMCOFactory,
    FixedMCOParameterFactory,
    ListedMCOParameterFactory,
    RangedMCOParameterFactory,
    RangedVectorMCOParameterFactory,
    CategoricalMCOParameterFactory,
)

from .ng_mco_model import NevergradMCOModel
from .ng_mco import NevergradMCO
from .ng_mco_communicator import NevergradMCOCommunicator


class NevergradMCOFactory(BaseMCOFactory):
    """ Base NevergradMCO Factory with generic configuration.
    Users might want to add custom MCOCommunicator instead of the
    BaseMCOCommunicator. Also, the parameter_factory method can
    be updated for custom MCOParameterFactories.
    """

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
        return NevergradMCOCommunicator

    #: Factory classes of the parameters the MCO supports.
    def get_parameter_factory_classes(self):
        return [
            FixedMCOParameterFactory,
            ListedMCOParameterFactory,
            RangedMCOParameterFactory,
            CategoricalMCOParameterFactory,
            RangedVectorMCOParameterFactory
        ]
