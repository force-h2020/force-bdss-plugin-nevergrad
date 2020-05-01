import math
import numpy as np

from force_bdss.mco.parameters.mco_parameters import (
    FixedMCOParameter,
    RangedMCOParameter,
    RangedVectorMCOParameter,
    ListedMCOParameter,
    CategoricalMCOParameter,
)

from force_bdss.core.kpi_specification import (
    KPISpecification
)


class TwoMinimaObjective:
    """ An objective function consisting of two Gaussian minima.
    """

    def get_params(self):
        """ Get the MCO parametrization.

        Return
        ------
        list of MCOParameter
            The MCO parametrization.
        """

        xycoordinates = RangedVectorMCOParameter(
            factory=None,
            dimension=2,
            lower_bound=[-2.0, -2.0],
            upper_bound=[2.0, 2.0],
            initial_value=[-0.5, -0.5]
        )
        return [xycoordinates]

    def get_kpis(self):
        """ Get the kpi (objective) specification.

        Return
        ------
        list of  KPISpecification.
            The kpi specification.
        """

        kpis = [
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

        return kpis

    def objective(self, p):
        """ The objective function.

        Parameters
        ----------
        p: list of float or list
            The parameter values in the MCO format.

        Return
        ------
        numpr array
            The kpi (objective) values.
        """
        # x-y coordinates (these are two-dimensional Gaussians)
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
        """ Get the psoition of the global minimum.

        Notes
        -----
        The global minimum is the first Gaussian, with its centre at [-1, -1].
        """
        return [[-1.0, -1.0]], 1

    def is_pareto_optimal(self, p):
        """ Check if a point is a member of the Pareto-set.

        Parameters
        ----------
        p: list of float or list
            The point's parameter values in the MCO format.

        Return
        ------
        bool
            If the point is a member of the Pareto-set.

        Notes
        -----
        The Pareto set roughly extends is a straight line (as long
        as each Gaussian's is symmetric) between the centre's of
        the Gaussians. This roughly test's that: is the point within
        a bounding box that contains the centres within a border of 0.5.
        """
        xy = p[0]
        for i in range(2):
            if (xy[i] < -1.5) or (xy[i] > 1.5):
                return False
        return True
