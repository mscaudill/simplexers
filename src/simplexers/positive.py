""" """

import numpy as np
import numpy.typing as npt

from simplexers import arraytools

def positive_simplexer(y: npt.NDArray, s: int = 1, axis: int = -1):
    """Computes the Euclidean projection of each 1-D array in y along axis onto
    the positive simplex.

    The simplex projection solves the following constraint problem for a 1-D
    array 'y':

    minimize 0.5 * ||x - y||**2 subject to sum(x_i) = s and 0 <= x_i.

    In words, we seek a vector x that is closest to y but must be in the
    positive orthant and lie on the hyperplane x.T * np.ones = s. This function
    implements the sort method of reference 1 to obtain an exact solution.

    Args:
        y:
            A 1-D or 2-D array, of vector(s) to project onto a simplex.
        s:
            A parameter that controls the position of the hyperplane in which
            the resultant vector(s) must lie. Defaults to 1 which creates
            a projection onto the probability simplex.
        axis:
            The axis of y .

    References:
        1. Efficient Learning of Label Ranking by Soft Projections onto Polyhedra.
           Shalev-Shwartz, S. and Singer, Y.  Journal of Machine Learning Research
           7 (2006).
        2. Large-scale Multiclass Support Vector Machine Training via Euclidean
           Projection onto the Simplex Mathieu Blondel, Akinori Fujino, and Naonori
           Ueda. ICPR 2014.
    """

    v = np.atleast_2d(y)
    # compute Lagrange multipliers 'thetas' (lemma 2 & 3 of Ref 1)
    mus = arraytools.slice_along_axis(np.sort(v, axis=axis), step=-1, axis=axis)
    css = np.cumsum(mus, axis=axis) - s
    indices = np.arange(1, y.shape[axis] + 1)
    indices = arraytools.redim(indices, css.shape, axis=axis)
    # mus descend so count_nonzeros to get rho
    rho = np.count_nonzero(mus - css / indices > 0, axis=axis, keepdims=True)
    thetas = np.take_along_axis(css, rho-1, axis=axis) / rho

    return np.maximum(v - thetas, 0)


