"""A simplexer that projects 1-D arrays onto the positive (uncapped) simplex.

Functions:
        positive_simplexer:
            Computes the Euclidean projection of 1-D arrays onto the positive
            simplex.
"""

from typing import Callable, Dict

import numpy as np
import numpy.typing as npt

from simplexers.core import arraytools


def _sorting_simplexer(arr: npt.NDArray, s: float) -> npt.NDArray:
    """Computes the Euclidean projection of each 1-D array in y along axis onto
    the positive simplex using the sorting algorithm of Ref. 1.

    This method is O(n*log(n)) due to sorting and is therefore suitable for
    large component vectors.

    Args:
        arr:
            A 2-D array, of vector(s) to project onto a simplex. It is assumed
            each vectors components lie along axis = 1.
        s:
            A parameter that controls the position of the hyperplane in which
            the resultant vector(s) must lie. A value of 1 forces all components
            to be in [0,1] and sum(x_i) = 1 which is the standard probability
            simplex.
    Returns:
        A 2-D array of vector projections one per vector in arr.

    References:
        1. Efficient Learning of Label Ranking by Soft Projections onto Polyhedra.
           Shalev-Shwartz, S. and Singer, Y.  Journal of Machine Learning Research
           7 (2006).
        2. Large-scale Multiclass Support Vector Machine Training via Euclidean
           Projection onto the Simplex Mathieu Blondel, Akinori Fujino, and Naonori
           Ueda. ICPR 2014.
    """

    v = np.atleast_2d(arr)
    axis = -1
    # compute Lagrange multipliers 'thetas' (lemma 2 & 3 of Ref 1)
    mus = arraytools.slice_along_axis(np.sort(v, axis=axis), step=-1, axis=axis)
    css = np.cumsum(mus, axis=axis) - s
    indices = np.arange(1, arr.shape[axis] + 1)
    indices = arraytools.redim(indices, css.shape, axis=axis)
    # mus descend so count_nonzeros to get rho
    rho = np.count_nonzero(mus - css / indices > 0, axis=axis, keepdims=True)
    thetas = np.take_along_axis(css, rho - 1, axis=axis) / rho

    result: npt.NDArray = np.maximum(v - thetas, 0)
    return result


def positive_simplexer(
    arr: npt.NDArray,
    s: float,
    axis: int = -1,
    method: str = 'sort',
    **kwargs,
) -> npt.NDArray:
    """Computes the Euclidean projection of each 1-D array in y along axis onto
    the positive simplex.

    The positive simplex projection locates the vector x closest to vector
    y subject to a positivity constraint and sum constraint:

    minimize 0.5 * ||x - y||**2 subject to 0 <= x_i and sum(x_i) = s

    In words, we seek a vector x that is closest to y but must be in the
    positive orthant and lie on the hyperplane x.T * np.ones = s.
    Args:
        arr:
            A 2-D array, of vector(s) to project onto a simplex.
        s:
            A parameter that controls the position of the hyperplane in which
            the resultant vector(s) must lie. A value of 1 forces all components
            to be in [0,1] and sum(x_i) = 1 which is the standard probability
            simplex.
        axis:
            The axis of arr containing the components to project onto the
            positive simplex .

    References:
        1. Efficient Learning of Label Ranking by Soft Projections onto Polyhedra.
           Shalev-Shwartz, S. and Singer, Y.  Journal of Machine Learning Research
           7 (2006).
        2. Large-scale Multiclass Support Vector Machine Training via Euclidean
           Projection onto the Simplex Mathieu Blondel, Akinori Fujino, and Naonori
           Ueda. ICPR 2014.
    """

    if arr.ndim > 2:
        msg = 'Array(s) to project must have at most 2 dimensions'
        raise ValueError(msg)

    # more methods such as bisection will later be added
    # duplicates capped_simplexer but refactoring reduces clarity
    # pylint: disable=duplicate-code
    methods: Dict[str, Callable] = {'sort': _sorting_simplexer}
    algorithm = methods[method]

    z = np.atleast_2d(arr)
    z = z.T if axis == 0 else z
    result: npt.NDArray = algorithm(z, s, **kwargs)
    # pylint: enable=duplicate-code

    return result.T if axis == 0 else result
