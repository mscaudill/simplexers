"""Module of tools for manipulating the size and values of ndarrays."""

import numpy as np

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

