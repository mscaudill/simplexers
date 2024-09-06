"""Module of tools for manipulating the size and values of ndarrays."""

from typing import Sequence

import numpy as np
import numpy.typing as npt


def is_array1D(x: npt.NDArray):
    """Returns True if x is an ndarray type and has one dim."""

    return True if isinstance(x, np.ndarray) and x.ndim == 1 else False

def is_array2D(x: npt.NDArray):
    """Returns True if x is an ndarray type and has two dim."""

    return True if isinstance(x, np.ndarray) and x.ndim == 2 else False

def normalize_axis(axis, ndim):
    """Returns a positive axis index for a supplied axis index of an ndim
    array.

    Args:
        axis:
            An positive or negative integer axis index.
        ndim:
            The number of dimensions to normalize axis by.
    """

    axes = np.arange(ndim)
    return axes[axis]

def pad_along_axis(arr, pad, axis=-1, **kwargs):
    """Wrapper for numpy pad allowing before and after padding along
    a single axis.

    Args:
        arr (ndarray):              ndarray to pad
        pad (int or array-like):    number of pads to apply before the 0th
                                    and after the last index of array along
                                    axis. If int, pad number of pads will be
                                    added to both
        axis (int):                 axis of arr along which to apply pad.
                                    Default pads along last axis.
        **kwargs:                   any valid kwarg for np.pad
    """

    #convert int pad to seq. of pads & place along axis of pads
    pad = [pad, pad] if isinstance(pad, int) else pad
    pads = [(0,0)] * arr.ndim
    pads[axis] = pad
    return np.pad(arr, pads, **kwargs)

def slice_along_axis(arr, start=None, stop=None, step=None, axis=-1):
    """Returns slice of arr along axis from start to stop in 'step' steps.

    (see scipy._arraytools.axis_slice)

    Args:
        arr (ndarray):              an ndarray to slice
        start, stop, step (int):    passed to slice instance
        axis (int):                 axis of array to slice along

    Returns: sliced ndarray
    """

    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(start, stop, step)
    return arr[tuple(slicer)]

def redim(x: npt.NDArray, shape: Sequence, axis: int):
    """Increases the dimensionality of a 0D or 1D array by embedding x into
    a ndim array whose shape is one along all axes except the embedding axis.

    This is a generic form of numpy's broadcast_to that does not force the
    embedding axis to match the axis in the output shape (see example)

    Args:
        x:
            A 0D or 1D array whose dimensions will be increased.
        shape:
            The shape of the new array in which x will be embedded
        axis:
            The axis along which x will be embedded.

    Examples:
        >>> import numpy as np
        >>> x = np.array([1,2,3,4])
        >>> # broadcast_to works when trailing dimensions match
        >>> y = np.broadcast_to(x, (2, 4))
        >>> # and fail when they do not
        >>> np.broadcast_to(x, (4, 2)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: operands could not be broadcast together...
        >>> #redim works in both cases by violating broadcast rules
        >>> redim(x, (4, 2), axis=0)
        array([[1],
               [2],
               [3],
               [4]])

    Returns:
        An ndarray with shape 1 along all axes except axis whose length matches
        x.

    Raises:
        A ValueError is raised if x is not a 0D or 1D array.
    """

    if x.ndim > 1:
        msg = 'x must be a zero or 1-dimensional array'
        raise ValueError(msg)

    ax = normalize_axis(axis, len(shape))
    outshape = np.ones(len(shape), int)
    outshape[ax] = len(x)

    return x.reshape(*outshape)

