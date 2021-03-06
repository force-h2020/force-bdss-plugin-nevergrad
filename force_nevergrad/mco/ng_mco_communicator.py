#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

import sys
import re

from force_bdss.api import (
    BaseMCOCommunicator,
    DataValue,
    FixedMCOParameter,
    RangedMCOParameter,
    ListedMCOParameter,
    CategoricalMCOParameter
)


class NevergradMCOCommunicator(BaseMCOCommunicator):
    """ Command-line evaluation.

    Examples
    --------
    Within an EDM environment:
    Evaluate the point (-1.0, 1.0) and write to output.txt,

    echo 1.0,-1.0 | force_bdss --evaluate gaussian.json > output.txt

    Evaluate the point on the first line of input.txt and write to output.txt,

    Windows-native:
    type input.txt | force_bdss --evaluate gaussian.json > output.txt

    Mac/bash:
    cat input.txt | force_bdss --evaluate gaussian.json > output.txt

    When running outside the shell and its environment, prefix force-bdss
    with: edm run -e environment-name --

    Notes
    -----
    Evaluate a single point in parameter space, from stdin, and return the KPIs
    to stdout.
    If you want to evaluate multiple points, either:
    1) Write a `BaseMCO` that employs a `SubprocessWorkflow` to broadcast
    points as stdin and waits for KPIs to return as stdout.
    2) Write a bash pipe that iterates through sets of single points and
    processes the output accordingly.
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
            elif isinstance(param, ListedMCOParameter):
                v = param.levels[0]
            elif isinstance(param, CategoricalMCOParameter):
                v = param.categories[0]
            else:
                v = 0.0

            # if the parameter value is specified in the stdin, then take this
            # and attempt to convert it from string into the appropriate type.
            try:
                v = type(v)(data[i])
            except(ValueError, IndexError):
                pass
            inputs.append(DataValue(value=v, name=param.name, type=param.type))

        return inputs

    def send_to_mco(self, model, kpi_results):
        """ Output the KPIs to stdout.
        """
        # tab-delimited output
        data = "\t".join([str(dv.value) for dv in kpi_results]) + '\n'
        sys.stdout.write(data)
