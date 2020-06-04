#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.
import sys
import re

from force_bdss.api import (
    BaseMCOCommunicator,
    DataValue,
    FixedMCOParameter,
    RangedMCOParameter,
    RangedVectorMCOParameter,
    ListedMCOParameter,
    CategoricalMCOParameter
)


class NevergradMCOCommunicator(BaseMCOCommunicator):
    """ Command-line evaluation.

    Examples
    --------
    Evaluate the point (-1.0, 1.0) and write to output.txt,

    echo 1.0,-1.0 | edm run -e force-py36 -- force_bdss
    --evaluate gaussian.json > output.txt

    Evaluate the point on the first line of input.txt and write to output.txt,

    Windows-native:
    type input.txt | edm run -e force-py36 -- force_bdss
    --evaluate gaussian.json > output.txt

    Mac/bash:
    cat input.txt | edm run -e force-py36 -- force_bdss
    --evaluate gaussian.json > output.txt

    If you are in the EDM force-py36 environment, then just leave out the
    'edm run -e force-py36 --' part of the command.

    Notes
    -----
    Can only evaluate a single point from stdin. If you want to evaluate
    multiple points listed in a file, then you must write a script
    that loops through the file's lines (one line per point) and evaluates
    each one in turn. cf. evalf.cmd for a windows-native script.
    """

    def receive_from_mco(self, model):
        """ Get the parameter values to evaluate from stdin.
        """
        # Read in a line of points to evaluate.
        # Can be tab or comma delimited.
        line = sys.stdin.readline()
        data = re.split(r'[,\s]+', line)

        # Get the parameter values based on parameterization and stdin
        inputs = []
        for i, param in enumerate(model.parameters):

            # default value of parameter
            if isinstance(param, FixedMCOParameter):
                v = param.value
            elif isinstance(param, RangedMCOParameter):
                v = param.initial_value
            elif isinstance(param, RangedVectorMCOParameter):
                v = param.initial_value
            elif isinstance(param, ListedMCOParameter):
                v = param.levels[0]
            elif isinstance(param, CategoricalMCOParameter):
                v = param.categories[0]
            else:
                v = 0.0

            # if the parameter value is specified in the stdin, then take this
            # and attempt to convert it from string into the appropriate type.
            q = v
            try:
                q = (type(v)(data[i]), v)[i >= len(data)]
            except ValueError:
                pass

            # append input
            inputs.append(DataValue(value=q, name=param.name))

        return inputs

    def send_to_mco(self, model, kpi_results):
        """ Output the KPIs to stdout.
        """
        # tab-delimited output
        data = "\t".join([str(dv.value) for dv in kpi_results]) + '\n'
        sys.stdout.write(data)
