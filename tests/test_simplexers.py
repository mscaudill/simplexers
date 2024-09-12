"""A module for testing positive and capped simplexers.

Typical usage example:
    >>> #run all test
    >>> !pytest test_simplexers
    >>> # test specific function
    >>> !pytest test_simplexers::test_positive_sums
"""

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from simplexers import positive, capped

@pytest.fixture(scope='module')
def rng():
    """Returns a reusable numpy default_rng object for creating reproducible but
    random arrays."""

    seed = 0
    return np.random.default_rng(seed)

@pytest.fixture(scope='module', params=range(10))
def size(rng, request):
    """A fixture that yields param count 2-tuple shapes."""

    # the minimum and maximum len along the 2 axes
    low, high = 3, 100
    yield tuple(rng.integers(low=low, high=high, size=2))

@pytest.fixture(scope='module', params=range(1,5))
def extrema(request):
    """A fixture that yields 2-tuples (0, param) of extreme values."""

    yield (0, request.param)

@pytest.fixture(scope='module')
def random2D(rng, size, extrema):
    """A fixture that yields random arrays drawn from a uniform distribution
    with min and max extrema and size."""

    return rng.uniform(*extrema, size)

@pytest.mark.parametrize('arr', [lazy_fixture('random2D')])
@pytest.mark.parametrize('s', [1, 2, 3])
@pytest.mark.parametrize('axis', [0, 1])
def test_positive_sums(arr, s, axis):
    """Validates that a projection onto the positive simplex using the
    positive_simplexer satisfies the sum constraint s."""

    projection = positive.positive_simplexer(arr, s=s, axis=axis)
    assert np.allclose(np.sum(projection, axis=axis), s)

@pytest.mark.parametrize('arr', [lazy_fixture('random2D')])
@pytest.mark.parametrize('s', [1, 2, 3])
@pytest.mark.parametrize('axis', [0, 1])
def test_capped_sums_root(arr, s, axis):
    """Validates that a projection onto the capped simplex using the root method
    of the capped_simplexer satisfies the sum constraint."""

    projection = capped.capped_simplexer(arr, s=s, axis=axis)
    assert np.allclose(np.sum(projection, axis=axis), s)

@pytest.mark.parametrize('arr', [lazy_fixture('random2D')])
@pytest.mark.parametrize('s', [1, 2, 3])
@pytest.mark.parametrize('axis', [0, 1])
def test_capped_sums_sort(arr, s, axis):
    """Validates that a projection onto the capped simplex using the sort method
    of the capped simplexer satisfies the sum constraint."""

    projection = capped.capped_simplexer(arr, s=s, axis=axis, method='sort')
    assert np.allclose(np.sum(projection, axis=axis), s)

@pytest.mark.parametrize('arr', [lazy_fixture('random2D')])
@pytest.mark.parametrize('s', [1, 2, 3])
@pytest.mark.parametrize('axis', [0, 1])
def test_capped_components_root(arr, s, axis):
    """Validates that a projection onto the capped simplex using the root method
    of the capped_simplexer satisfies the component constraint."""

    projection = capped.capped_simplexer(arr, s=s, axis=axis)
    assert np.all(np.logical_and(projection >= 0, projection <= 1))

@pytest.mark.parametrize('arr', [lazy_fixture('random2D')])
@pytest.mark.parametrize('s', [1, 2, 3])
@pytest.mark.parametrize('axis', [0, 1])
def test_capped_components_sort(arr, s, axis):
    """Validates that a projection onto the capped simplex using the sort method
    of the capped_simplexer satisfies the component constraint."""

    projection = capped.capped_simplexer(arr, s=s, axis=axis, method='sort')
    assert np.all(np.logical_and(projection >= 0, projection <= 1))


@pytest.mark.parametrize('arr', [lazy_fixture('random2D')])
@pytest.mark.parametrize('axis', [0, 1])
def test_projection_agreement(arr, axis):
    """Validates that all projections agree when the sum constraint is 1.

    When the sum constraint is 1 the positive simplex and the capped simplex
    should yield the same projections.
    """

    positive_projection = positive.positive_simplexer(arr, s=1, axis=axis)
    root_projection = capped.capped_simplexer(arr, axis=axis, method='root')
    sort_projection = capped.capped_simplexer(arr, axis=axis, method='sort')

    assert np.allclose(positive_projection, root_projection)
    assert np.allclose(positive_projection, sort_projection)

@pytest.mark.parametrize('arr', [lazy_fixture('random2D')])
@pytest.mark.parametrize('s', [1, 2, 3])
@pytest.mark.parametrize('axis', [0, 1])
def test_projection_agreement(arr, axis, s):
    """Validates that root and sort capped projections agree."""

    root_proj = capped.capped_simplexer(arr, s=s, axis=axis, method='root')
    sort_proj = capped.capped_simplexer(arr, s=s, axis=axis, method='sort')

    assert np.allclose(root_proj, sort_proj)

def test_shape_error(rng):
    """Validates that a ValueError is raised if the input arr to
    a positive_simplexer or capped simplexer has more than 2 dims."""

    arr = rng.uniform(size=(3, 10, 100))
    with pytest.raises(ValueError):
        positive.positive_simplexer(arr, s=1, axis=1)
        capped.capped_simplexer(arr, s=1, axis=1)
