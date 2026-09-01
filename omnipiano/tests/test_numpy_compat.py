"""Cross-version regression tests for the NumPy ``copy=False`` shim."""

from __future__ import annotations

import numpy as np

from omnipiano.envs.registration import _np_array_compat


class _ArraySubclass(np.ndarray):
    pass


def test_copy_false_shim_accepts_subok_and_ndmin() -> None:
    source = np.arange(3, dtype=np.float64).view(_ArraySubclass)

    preserved = _np_array_compat(
        source,
        copy=False,
        subok=True,
        ndmin=2,
    )
    stripped = _np_array_compat(source, copy=False, subok=False)

    assert isinstance(preserved, _ArraySubclass)
    assert preserved.shape == (1, 3)
    assert type(stripped) is np.ndarray
    assert np.shares_memory(preserved, source)
    assert np.shares_memory(stripped, source)


def test_broadcast_to_call_path_works_with_global_shim() -> None:
    value = np.array([1.0, 2.0])
    broadcast = np.broadcast_to(value, shape=(3, 2), subok=True)
    assert broadcast.shape == (3, 2)
    assert np.array_equal(broadcast[0], value)
