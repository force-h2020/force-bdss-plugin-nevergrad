#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

import math
import numpy as np
from abc import abstractmethod

from force_bdss.api import (
    RangedVectorMCOParameter,
    ListedMCOParameter,
    KPISpecification
)


class BaseObjective:
    """ An objective function for the unit-testing of optimizers.
    """

    @abstractmethod
    def get_params(self):
        """ Get the MCO parametrization.

            Return
            ------
            list of MCOParameter
                The MCO parametrization.
        """

    @abstractmethod
    def get_kpis(self):
        """ Get the kpi (objective) specification.

            Return
            ------
            list of  KPISpecification.
                The kpi/objective specification.
        """

    @abstractmethod
    def objective(self, p):
        """ The objective function.

        Parameters
        ----------
        p: list of float/string/list
            The parameter values in the MCO format.

        Return
        ------
        numpy array
            The kpi/objective values.
        """

    @abstractmethod
    def get_global_optimum(self):
        """ Get the position of the global minimum.

        Returns
        -------
        list of float/string/list
            The parameters at the global minimum
        float
            The tolerance (distance) that an optimizer should
            reasonably be able to get to (for each numerical parameter)
        """

    @abstractmethod
    def is_pareto_optimal(self, p):
        """ Check if a point is a member of the Pareto-set.

        Parameters
        ----------
        p: list of float/string/list
            The point's parameter values.

        Return
        ------
        bool
            If the point is a member of the Pareto-set.
        """


class GridValleyObjective(BaseObjective):
    """ A valley formed by two opposing exponentials (each an objective)
    over a grid of points on the x-y plane.
    Useful for:
    Illustrating optimization with non-continuous parameters.
    """

    def __init__(self):

        self.xpoints = [i / 10 for i in range(11)]
        self.ypoints = [i / 10 for i in range(11)]

    def get_params(self):

        return [
            ListedMCOParameter(
                name="x",
                factory=None,
                levels=self.xpoints
            ),
            ListedMCOParameter(
                name="y",
                factory=None,
                levels=self.ypoints
            )
        ]

    def get_kpis(self):

        return [
            KPISpecification(
                name="slope1",
                objective="MINIMISE",
                auto_scale=False,
                scale_factor=1.0
            ),
            KPISpecification(
                name="slope2",
                objective="MINIMISE",
                auto_scale=False,
                scale_factor=1.0
            )
        ]

    def objective(self, p):

        # x,y coordinates (grid point)
        x = p[0]
        y = p[1]

        # angle of valley (radians)
        # The valley bottom is a straight line across the xy plane.
        # The global minimum is the point on the grid closest to this line.
        # As long as the angle is an improper fraction of pi, there should
        # only be one global minima.
        alpha = math.pi / 2.9

        # rotated x coordinate (~distance from valley centre)
        x_r = x * math.cos(-alpha) - y * math.sin(-alpha)

        # exponential slopes
        z1 = math.exp(-0.5 - x_r)
        z2 = math.exp(x_r - 1.5)

        return np.array([z1, z2])

    def get_global_optimum(self):

        # find this numerically.
        # There is no doubt a simple analytic expression for the
        # line of the valley bottom, but I'll leave that to someone else.
        i = 0
        j = 0
        for x in self.xpoints:
            for y in self.ypoints:
                if sum(self.objective([x, y])) < sum(self.objective([i, j])):
                    i = x
                    j = y

        return [i, j], 0.05

    def is_pareto_optimal(self, p):
        return True


class TwoMinimaObjective(BaseObjective):
    """ Two Gaussian minima (each an objective) on the x-y plane.
    Useful for:
    Seeing if an optimizer can find the global minima (deeper of
    the two Gaussians) versus a local (the other Gaussian).
    Has an obvious Pareto-set stretching between the two-mimima.
    """

    def get_params(self):

        return [
            RangedVectorMCOParameter(
                name="xy_coordinates",
                factory=None,
                dimension=2,
                lower_bound=[-2.0, -2.0],
                upper_bound=[2.0, 2.0],
                initial_value=[-0.5, -0.5]
            )
        ]

    def get_kpis(self):

        return [
            KPISpecification(
                name="gauss1",
                objective="MAXIMISE",
                auto_scale=False,
                scale_factor=1.0
            ),
            KPISpecification(
                name="gauss2",
                objective="MINIMISE",
                auto_scale=False,
                scale_factor=1.0
            )
        ]

    def objective(self, p):

        # x-y coordinates
        xy = p[0]
        # centre of each Gaussian
        cent = [[-1.0, -1.0], [1.0, 1.0]]
        # width of each Gaussian
        sigma = [0.4, 0.2]
        # peak of each Gaussian
        # NOTE: peak 1 is maximised, peak 2 is minimized
        peak = [4.0, -1.0]
        # objectives (amplitudes of Gaussian components)
        objective = [0.0, 0.0]

        for j in range(2):
            for i in range(2):
                objective[j] += \
                    ((xy[i] - cent[j][i])**2.0)/(2.0*sigma[j]**2.0)
            objective[j] = peak[j]*math.exp(-objective[j])

        return np.array(objective)

    def get_global_optimum(self):
        """ The global minimum is the first Gaussian,
        with its centre at [-1, -1].
        """
        return [[-1.0, -1.0]], 0.1

    def is_pareto_optimal(self, p):
        """ The Pareto set roughly extends is a straight line (as long
        as each Gaussian's is symmetric) between the centre's of
        the Gaussians. This roughly test's that: is the point within
        a bounding box that contains the centres within a border of 0.5.
        """
        xy = p[0]
        for i in range(2):
            if (xy[i] < -1.5) or (xy[i] > 1.5):
                return False
        return True
