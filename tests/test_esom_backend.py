"""ESOM backend parameter."""

import pytest

from pyesom.projection.esom import ESOM


def test_backend_defaults_to_minisom():
    som = ESOM(4, 5, 3)
    assert som.backend == "minisom"


def test_backend_explicit_minisom():
    som = ESOM(4, 5, 3, backend="minisom")
    assert som.backend == "minisom"


def test_backend_aliases_normalized():
    som = ESOM(4, 5, 3, backend="MiniSom")
    assert som.backend == "minisom"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unsupported backend"):
        ESOM(4, 5, 3, backend="torchsom")
