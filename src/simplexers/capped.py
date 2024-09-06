"""A simplexer that projects 1-D arrays onto the s-capped simplex.

Functions:
    capped_simplexer
"""

from typing import Optional
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import optimize

from simplexers import arraytools
from simplexers.positive import positive_simplexer


def _sort_simplexer(arr: npt.NDArray, s: int) -> npt.NDArray:
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

def _root_simplexer(
    arr: npt.NDArray,
    s: int,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> npt.NDArray:
    """Computes the Euclidean projection of each 1-D arr in arr onto the
    s-capped simplex via the Lagrangian root finding alorithm of Reference 1.

    This algorithm is O(n) and therefore suitable for vectors with large numbers
    of components ( > 50).

    Args:
        arr:
            A 2-D numpy array of vectors to project onto the s-capped simplex.
            It is assumed the axis of arr containing the vector elements to
            project is the last axis.
        s:
            The sum constraint of the simplex
        rng:
            A numpy random Generator instance for reproduing the root
            finding iterations. Default None uses a random seed of None for the
            generator.
        **kwargs:
            Any valid kwarg for scipy.optimize.newton root finding method.

    Returns:
        A 2-D array of projected vectors with shape matching arr.

    """

    # create initial gamma values for newton-raphson one per vector in arr
    rng = rng if rng else np.random.default_rng(None)
    maxes = np.max(arr, axis=1)
    mins = np.min(arr, axis=1)
    init_gammas = [rng.uniform(m - 1, m) for m in maxes]
    #init_gammas = [rng.uniform(mn-1 + 1e-1, mx-1e-1) for mn, mx in zip(mins,
    #    maxes)]
    print(init_gammas)

    def omega_prime(gamma, y, s):
        """Derivative of the projection Largrangian wrt gamma at the critical
        point x* for a 1-D vector y."""

        return s - np.sum(np.minimum(1, np.maximum(y - gamma, 0)))

    def omega_double_prime(gamma, y):
        """Second derivative of the projection Lagrangian wrt gamma at the critical
        point x* for a 1-D vector y."""

        v = y - gamma
        return np.count_nonzero(np.logical_and(v>0, v<1))

    # for each 1-D array perform newton-raphson to get gamma and x at saddle pt
    result = np.zeros_like(arr)
    for idx, (init_gamma, y) in enumerate(zip(init_gammas, arr)):

        func = partial(omega_prime, y=y, s=s)
        fprime = partial(omega_double_prime, y=y)
       
        # TODO
        # omega_double_prime can be zero so we need to readjust feasible gammas

        try:
            gamma_star = optimize.newton(func, init_gamma, fprime)
        except Exception as e:
            print(init_gamma)
            raise e
        result[idx] = np.minimum(1, np.maximum(y - gamma_star, 0))

    return result

def _optimal_method(ncomponents: int, s: int):
    """Returns the optimal projection method based on the number of components
    in the vectors to project and the sum constraint of the s-capped simplex. 

    Args:
        ncomponents:
            The number of elements of the vectors to be projected onto the
            capped simplex.
        s:
            The integer sum constraint of the capped simplex.

    Returns:
        A callable
    """

    if s == 1:
        return positive_simplexer

    if ncomponents <= 50:
        return _sort_simplexer

    return _root_simplexer

def capped_simplexer(
    arr: npt.NDArray,
    s: int = 1,
    axis: int = -1,
    method: str = None,
    **kwargs,
) -> npt.NDArray:
    """ """

    methods = {
            'sort': _sort_simplexer,
            'newton': _root_simplexer,
            'probability': positive_simplexer,
             }

    if method is None:
        algorithm = _optimal_method(arr.shape[-1], s)

    else:
        algorithm = methods[method]

    print(algorithm.__name__)

    if algorithm == positive_simplexer and s != 1:
        msg = "A probability simplexer requires the sum constraint to be one."
        raise ValueError(msg)

    z = np.atleast_2d(arr)
    z = z.T if axis != 0 else z

    result = algorithm(arr, s, **kwargs)
    return result.T if axis != 0 else result




if __name__ == '__main__':

    import time

    rng = np.random.default_rng(0)
    y = rng.random((50, 300)) - 0.5

    """
    t0 = time.perf_counter()
    target = positive_simplex(y)
    print('Positive Simplexer time: ', time.perf_counter() - t0)

    t0 = time.perf_counter()
    res_sort = _sort_simplexer(y, s=1)
    print('Capped Sorting Simplexer time ', time.perf_counter() - t0)

    t0 = time.perf_counter()
    res_root = _root_simplexer(y, s=1)
    print('Capped Root Simplexer time ', time.perf_counter() - t0)
    """

    t0 = time.perf_counter()
    result = capped_simplexer(y, method='newton', s=2)
    print(time.perf_counter() - t0)
