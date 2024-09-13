"""A simplexer that projects 1-D arrays onto the s-capped simplex.

Functions:
    capped_simplexer:
        Computes the Euclidean projection of 1-D array(s) onto the s-capped
        simplex.
"""

from functools import partial
from typing import Callable, Dict

import numpy as np
import numpy.typing as npt
from scipy import optimize

from simplexers.core import arraytools


def _sorting_simplexer(arr: npt.NDArray, s: int) -> npt.NDArray:
    """Computes the Euclidean projection of each 1-D array in arr onto the
    s-capped simplex via the sorting algorithm.

    This is the O(n**2) sorting algorithm of reference 1. It is suitable only
    for low dimensional vectors < ~100 elements. This function assumes the
    vector elements to be projected lie along the last axis and that the sum
    constraint is strictly less than the number of elements.

    Args:
        arr:
            A 2-D numpy array of vectors to project onto the s-capped simplex.
            It is assumed the axis of arr containing the vector elements to
            project is the last axis.
        s:
            The sum constraint of the simplex

    Returns:
        A 2-D array of projected vectors with shape matching arr.

    References:
        Projection onto the capped simplex. Weiran Wang and Canyi Lu.
        arXiv:1503.01002v1 [cs.LG]
    """

    # get the number of vector components and sort them
    n = arr.shape[1]
    sorting_idxs = np.argsort(arr, axis=-1)
    z = np.take_along_axis(arr, sorting_idxs, axis=-1)
    # compute the cumulative sums of the components padding with boundary cond.
    csums = np.cumsum(z, axis=1)
    z = arraytools.pad_along_axis(z, (0, 1), constant_values=np.inf)

    # for each vector compute the a,b partition that satisfies KKT conditions
    result = np.zeros_like(arr)
    for idx, (y, csum) in enumerate(zip(z, csums)):

        for a in range(0, n):
            # y[a-1] is -np.inf and csum[a-1] is 0
            low = -np.inf if a == 0 else y[a - 1]
            low_csum = 0 if a == 0 else csum[a - 1]

            for b in range(a + 1, n + 1):
                gamma = (s + b - n - csum[b - 1] + low_csum) / (b - a)

                conditions = [
                    low + gamma <= 0,
                    y[a] + gamma > 0,
                    y[b - 1] + gamma < 1,
                    y[b] + gamma >= 1,
                ]

                if all(conditions):
                    break

            else:
                # continue on to next a
                continue

            # an a and b > a was found
            break

        else:
            # no b > a so a == b == n - s
            a = b = n - s

        # add the sorted components to projection then unsort
        proj = np.zeros(n)
        proj[a:b] = y[a:b] + gamma
        proj[b:] = 1
        result[idx, sorting_idxs[idx]] = proj

    return result


def _root_simplexer(arr: npt.NDArray, s: float, **kwargs) -> npt.NDArray:
    """Computes the Euclidean projection of each 1-D array in arr onto the
    s-capped simplex via the critical points of the Lagrangian.

    This is the O(n) critical point method of Ref. 1 except the Netwon-Raphson
    method has been replaced by Brent's root finding method. This robustifies
    the algorithm because there is no need for a second derivative and its
    associated zero-division problem in the non-feasible region. Please note
    that Brent's method requires a bracketing interval [a,b] such that w', the
    Lagrangian's derivative, has opposite signs at the interval's endpoints:
    sign(w'(a)) = -1 * sign(w'(b)). This is satisfied if the interval is chosen
    to be [min(vector) - 1, max(vector)] for each vector in arr (see Ref 1 Eqn
    6).

    Args:
        arr:
            A 2-D numpy array of vectors to project onto the s-capped simplex.
        s:
            The sum constraint for each vector.
        kwargs:
            Any valid keyword only arguments for scipy.optimize.brentq function

    Returns:
        A 2-D array of vector projections one per vector in arr.
    """

    gamma_lows = np.min(arr, axis=1) - 1
    gamma_highs = np.max(arr, axis=1)
    brackets = np.stack((gamma_lows, gamma_highs)).T

    def omega_prime(gamma, y, s):
        """Derivative of the projection Largrangian wrt gamma at the critical
        point x* for a 1-D vector y."""

        return s - np.sum(np.minimum(1, np.maximum(y - gamma, 0)))

    result = np.zeros_like(arr)
    for idx, (bracket, y) in enumerate(zip(brackets, arr)):
        func = partial(omega_prime, y=y, s=s)
        gamma_star = optimize.brentq(func, *bracket, **kwargs)
        result[idx] = np.minimum(1, np.maximum(y - gamma_star, 0))

    return result


def capped_simplexer(
    arr: npt.NDArray,
    s: int = 1,
    axis: int = -1,
    method: str = 'root',
    **kwargs,
) -> npt.NDArray:
    """Computes the Euclidean projection of each 1-D array in arr onto the
    s-capped simplex.

    The s-capped simplex projection locates the vector x closest to vector
    y subject to a component constraint and sum constraint:

    minimize {||x - y||**2 subject to 0 <= x_i <= 1 and sum(x_i) = s}

    This method finds this solution by locating the Lagrangian's critical points
    (Ref. 1) or the sort algorithm (Ref. 2).

    Args:
        arr:
            A 2-D numpy array of vectors to project onto the s-capped simplex.
        s:
            The sum constraint for each vector.
        axis:
            The axis of arr containing vector components to project onto the
            simplex.
        method:
            A string method name specifying the algorithm used to make the
            projection. Must be one of {'root', 'sort'}. The 'root' method finds
            the roots of the derivative of the Lagrangian using Brents method.
            This method is O(n) where n is the number of vector components to
            project (See Ref. 1). 'Sort' uses the O(n**2) sorting algorithm
            (Ref. 2). For vectors with very few components it will have a speed
            advantage.
        kwargs:
            Any valid keyword only arguments for scipy.optimize.brentq function

    Returns:
        A 2-D array of vector projections one per vector in arr.

    Raises:
        A ValueError is issued if the input arr is not 1D or 2D.

    References:
        1. Andersen Ang, Jianzhu Ma, Nianjun Liu, Kun Huang, Yijie Wang, Fast
           Projection onto the Capped Simplex with Applications to Sparse
           Regression in Bioinformatics. arXiv:2110.08471 [math.OC]
        2. Projection onto the capped simplex. Weiran Wang and Canyi Lu.
           arXiv:1503.01002v1 [cs.LG]
    """

    if arr.ndim > 2:
        msg = 'Array(s) to project must have at most 2 dimensions'
        raise ValueError(msg)

    methods: Dict[str, Callable] = {
        'sort': _sorting_simplexer,
        'root': _root_simplexer,
    }
    algorithm = methods[method]

    z = np.atleast_2d(arr)
    z = z.T if axis == 0 else z
    result: npt.NDArray = algorithm(z, s, **kwargs)

    return result.T if axis == 0 else result
