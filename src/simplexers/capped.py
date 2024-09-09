"""A simplexer that projects 1-D arrays onto the s-capped simplex.

Functions:
    capped_simplexer
"""

from typing import Optional
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import optimize

from simplexers.core import arraytools
from simplexers import positive


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

    # FIXME need to verify operation if a == b for a vector!
    # get the number of vector components and sort them
    n = arr.shape[1]
    sorting_idxs = np.argsort(arr, axis=-1)
    z = np.take_along_axis(arr, sorting_idxs, axis=-1)
    # compute the cumulative sums of the components padding with boundary cond.
    csums = np.cumsum(z, axis=1)
    z = arraytools.pad_along_axis(z, (0,1), constant_values=np.inf)

    # for each vector compute the a,b partition that satisfies KKT conditions
    result = np.zeros_like(arr)
    for idx, (y, csum) in enumerate(zip(z, csums)):

        for a in range(0, n):
            # the a-1 boundary condition is -np.inf
            low = -np.inf if a==0 else y[a-1]

            for b in range(a+1, n+1):
                gamma = (s + b - n - csum[b-1] + csum[a-1]) / (b - a)

                conditions = [
                    low + gamma <= 0,
                    y[a] + gamma > 0,
                    y[b-1] + gamma < 1,
                    y[b] + gamma >= 1,
                    ]

                if all(conditions):
                    break

            else:
                continue

            break

        # add the sorted components to projection then unsort
        proj = np.zeros(n)
        proj[a:b] = y[a:b] + gamma
        proj[b:] = 1
        result[idx, sorting_idxs[idx]] = proj

    return result

def _root_simplexer(arr: npt.NDArray, s: float, **kwargs):
    """ """

    gamma_lows = np.min(arr, axis=1) - 1
    gamma_highs = np.max(arr, axis=1)
    brackets = np.stack((gamma_lows, gamma_highs)).T

    def omega_prime(gamma, y, s):
        """Derivative of the projection Largrangian wrt gamma at the critical
        point x* for a 1-D vector y."""

        return s - np.sum(np.minimum(1, np.maximum(y - gamma, 0)))

    result = np.zeros_like(arr)
    for idx, (bracket, y) in enumerate(zip(brackets, arr)):
        #print(idx)
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
    """ """

    methods = {
            'sort': _sorting_simplexer,
            'root': _root_simplexer,
            'probability': positive._sorting_simplexer
            }
    if method == 'probability' and s != 1:
        msg = 'A capped probability simplex requires a sum constraint of s=1'
        raise ValueError(msg)

    algorithm = methods[method]

    if arr.ndim > 2:
        msg = 'Array(s) to project must have at most 2 dimensions'
        raise ValueError(msg)

    z = np.atleast_2d(arr)
    z = z.T if axis == 0 else z
    result = algorithm(z, s, **kwargs)

    return result.T if axis == 0 else result




if __name__ == '__main__':

    import time

    rng = np.random.default_rng(0)
    y = rng.random((100000)) - 0.5

    t0 = time.perf_counter()
    target = capped_simplexer(y, s=1, axis=-1, method='probability')
    print('Positive Simplexer time: ', time.perf_counter() - t0)

    """
    t0 = time.perf_counter()
    res_sort = capped_simplexer(y, s=1, axis=0, method='sort')
    print('Sorting Simplexer time ', time.perf_counter() - t0)
    """

    t0 = time.perf_counter()
    res_root = capped_simplexer(y, s=1, axis=-1, method='root')
    print('Root Simplexer Time: ', time.perf_counter() - t0)

    #print(np.allclose(res_sort, target))
    print(np.allclose(res_root, target))
