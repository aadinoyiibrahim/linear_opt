from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from linear_opt.core.ir import ModelBuilder, ObjSense, Sense, VarType, sum_over


@given(m=st.integers(1, 6), n=st.integers(1, 6), seed=st.integers(0, 10_000))
def test_sum_over_matches_numpy(m: int, n: int, seed: int) -> None:
    x = np.random.default_rng(seed).random((m, n))
    np.testing.assert_allclose(sum_over((m, n), axis=1) @ x.ravel(), x.sum(axis=1))
    np.testing.assert_allclose(sum_over((m, n), axis=0) @ x.ravel(), x.sum(axis=0))


def test_sum_over_rejects_bad_axis() -> None:
    with pytest.raises(ValueError, match="axis"):
        sum_over((2, 2), axis=2)


def test_builder_assembles_blocks_in_order() -> None:
    b = ModelBuilder("t", ObjSense.MAXIMIZE)
    x = b.add_vars("x", (2, 3), obj=np.arange(6).reshape(2, 3))
    y = b.add_vars("y", 2, vtype=VarType.BINARY, lb=-5, ub=7)
    c = b.add_constrs("link", [(x, sum_over((2, 3), axis=1)), (y, -np.eye(2))], Sense.LE, 0.0)
    b.set_objective_constant(4.0)
    lm = b.build()

    assert (lm.num_vars, lm.num_constrs, lm.num_nonzeros) == (8, 2, 8)
    assert y.slice == slice(6, 8)
    np.testing.assert_array_equal(lm.lb[y.slice], [0, 0])  # binaries clipped to [0, 1]
    np.testing.assert_array_equal(lm.ub[y.slice], [1, 1])
    assert lm.is_mip and lm.c0 == 4.0
    assert lm.var_names()[:2] == ["x[0,0]", "x[0,1]"] and lm.var_names()[-1] == "y[1]"
    assert lm.constr_names() == ["link[0]", "link[1]"]
    assert c.sense is Sense.LE
    np.testing.assert_array_equal(x.values(np.arange(8.0)), [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(x.index(), [[0, 1, 2], [3, 4, 5]])


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (lambda b: b.add_vars("x", 2), "duplicate"),
        (lambda b: b.add_vars("bad name", 2), "identifier"),
        (lambda b: b.add_vars("z", 0), "positive"),
        (lambda b: b.add_vars("z", 2, lb=3, ub=1), "lower bound"),
        (lambda b: b.add_vars("z", 2, obj=np.inf), "finite"),
    ],
)
def test_builder_rejects_invalid_variables(action, message: str) -> None:  # type: ignore[no-untyped-def]
    b = ModelBuilder("t")
    b.add_vars("x", 2)
    with pytest.raises(ValueError, match=message):
        action(b)


def test_builder_rejects_inconsistent_constraints() -> None:
    b = ModelBuilder("t")
    x = b.add_vars("x", 3)
    with pytest.raises(ValueError, match="columns"):
        b.add_constrs("c", [(x, np.ones((1, 2)))], Sense.LE, 1)
    y = b.add_vars("y", 2)
    with pytest.raises(ValueError, match="rows"):
        b.add_constrs("c", [(x, np.ones((1, 3))), (y, np.ones((2, 2)))], Sense.LE, 1)
    with pytest.raises(ValueError, match="no terms"):
        b.add_constrs("c", [], Sense.LE, 1)
    foreign = ModelBuilder("other").add_vars("x", 3)
    with pytest.raises(ValueError, match="not from this model"):
        b.add_constrs("c", [(foreign, np.ones((1, 3)))], Sense.LE, 1)


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="no variables"):
        ModelBuilder("t").build()


def test_duplicate_entries_are_summed() -> None:
    b = ModelBuilder("t")
    x = b.add_vars("x", 1)
    b.add_constrs("c", [(x, [[1.0]]), (x, [[2.0]])], Sense.GE, 3)
    assert b.build().A.toarray().tolist() == [[3.0]]
