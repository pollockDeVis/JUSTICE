"""
RBF Class
Adapted from: https://github.com/JazminZatarain/MUSEH2O/blob/main/rbf/rbf_functions.py

This is the RBF class. It uses the modified squared exponential kernel, which was found to be the best performing kernel in the following paper.
Zatarain-Salazar, Jazmin, Jan H. Kwakkel, and Mark Witvliet. 
"Evaluating the choice of radial basis functions in multiobjective optimal control applications.
" Environmental Modelling & Software (2023): 105889.
"""

import itertools
from platypus import Real

import numpy as np

# import numba
from scipy.spatial.distance import cdist


def original_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """

    # sum over inputs
    a = rbf_input[np.newaxis, :] - centers
    b = a**2
    c = radii**2
    rbf_scores = np.exp(-(np.sum(b / c, axis=1)))

    # n_rbf x n_output, n_rbf
    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)

    return output


def squared_exponential_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """

    # sum over inputs
    # a = rbf_input[np.newaxis, :] - centers
    a = cdist(rbf_input[np.newaxis, :], centers)
    b = a.T**2
    c = 2 * radii**2
    rbf_scores = np.exp(-(np.sum(b / c, axis=1)))

    # n_rbf x n_output, n_rbf
    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)

    return output


class RBF:
    def __init__(
        self, n_rbfs, n_inputs, n_outputs, rbf_function=squared_exponential_rbf
    ):
        self.n_rbfs = n_rbfs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.rbf = rbf_function

        types = []
        c_i = []
        r_i = []
        w_i = []
        count = itertools.count()
        for i in range(self.n_rbfs):
            for j in range(self.n_inputs):
                types.append(Real(-1, 1))  # center
                c_i.append(next(count))
                types.append(Real(0, 1))  # radius
                r_i.append(next(count))

        for _ in range(self.n_rbfs):
            for _ in range(self.n_outputs):
                types.append(Real(0, 1))  # weight
                w_i.append(next(count))  # weight

        self.platypus_types = types
        self.c_i = np.asarray(c_i, dtype=int)
        self.r_i = np.asarray(r_i, dtype=int)
        self.w_i = np.asarray(w_i, dtype=int)

        self.centers = None
        self.radii = None
        self.weights = None

    def get_shape(self):
        """
        This method returns the shapes of centers, radii, and weights if they have been initialized.

        Returns:
            tuple: A tuple containing the shapes of centers, radii, and weights.
        """
        if self.c_i is None or self.r_i is None or self.w_i is None:
            raise ValueError(
                "Centers, radii, and weights must be initialized before getting their shapes."
            )

        return self.c_i.shape, self.r_i.shape, self.w_i.shape

    def set_decision_vars(self, decision_vars):
        decision_vars = decision_vars.copy()

        self.centers = decision_vars[self.c_i].reshape((self.n_rbfs, self.n_inputs))
        self.radii = decision_vars[self.r_i].reshape((self.n_rbfs, self.n_inputs))
        self.weights = decision_vars[self.w_i].reshape((self.n_rbfs, self.n_outputs))

        # sum of weights per input is 1
        self.weights /= self.weights.sum(axis=0)[np.newaxis, :]

    def apply_rbfs(self, inputs):
        outputs = self.rbf(inputs, self.centers, self.radii, self.weights)

        return outputs
