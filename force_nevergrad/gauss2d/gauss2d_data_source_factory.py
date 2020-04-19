from force_bdss.api import BaseDataSourceFactory

from .gauss2d_data_source_model import Gauss2dDataSourceModel
from .gauss2d_data_source import Gauss2dDataSource


class ArrheniusParametersFactory(BaseDataSourceFactory):
    def get_identifier(self):
        return "gauss_2d"

    def get_name(self):
        return "2D gaussian"

    def get_model_class(self):
        return Gauss2dDataSourceModel

    def get_data_source_class(self):
        return Gauss2dDataSource
