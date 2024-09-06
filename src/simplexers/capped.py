"""

"""


import numpy as np
import numpy.typing as npt
from scipy import optimize

from ultrak.core import arraytools
from simplexers.arraytools import pad_along_axis, slice_along_axis


def _sort_simplexer(y: npt.NDArray, s: int) -> npt.NDArray:
    """ """

    # sort input saving the sorting indices
    sorting_idxs = np.argsort(y)
    z = np.take_along_axis(y, sorting_idxs, axis=-1)
    csums = np.cumsum(z, axis=1)
    csums = pad_along_axis(csums, (0, 1), axis=1, constant_values=np.inf)

    result = np.zeros_like(z)
    for idx, (arr, csum) in enumerate(zip(z, csums)):

        for a in range(0, n):
            # if a is at 0th the previous value is defined as -np.inf
            low = -np.inf if a==0 else arr[a-1]

            for b in range(a+1, n+1):
                gamma = (s + b - N - csum[b-1] + csum[a-1]) / (b - a)

                conditions = [
                    low + gamma <= 0,
                    arr[a] + gamma > 0,
                    arr[b-1] + gamma < 1,
                    arr[b] + gamma >= 1,
                    ]

                if all(conditions):
                    break

            else:
                continue

            break

        proj = np.zeros(n)
        proj[a:b] = arr[a:b] + gamma
        proj[b:] = 1
        result[idx, sorting_idxs[idx]] = proj

    return result


# FIXME remove the default s
def simplex(y: npt.NDArray, s: int = 1, axis: int = -1):
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



if __name__ == '__main__':

    import time

    rng = np.random.default_rng(0)
    y = rng.random((10, 100)) - 0.5

    t0 = time.perf_counter()
    target = simplex(y)
    print('Positive Simplexer time: ', time.perf_counter() - t0)

    t0 = time.perf_counter()
    res = _sort_simplexer(y, s=1)
    print('Capped Sorting Simplexer time ', time.perf_counter() - t0)
