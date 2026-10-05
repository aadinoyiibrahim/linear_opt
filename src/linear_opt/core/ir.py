r"""Solver-neutral intermediate representation (IR) of a linear / mixed-integer model.

Every model in this package is compiled to the canonical form

.. math::

    \min_x / \max_x \; c^\top x + c_0
    \quad\text{s.t.}\quad A x \;\{\le, =, \ge\}\; b,
    \quad \ell \le x \le u,
    \quad x_j \in \mathbb{Z} \;\; (j \in \mathcal{I}).

Problems never talk to a solver directly. They declare *blocks* of variables
and constraints through :class:`ModelBuilder`; each backend (Gurobi, HiGHS)
translates the resulting :class:`LinearModel` with its native matrix API. Blocks
keep the algebra readable: a transport model has one ``flow`` block of shape
``(m, n)`` and constraint blocks ``supply`` and ``demand``, and solution values or
duals are read back by block rather than by raw index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ArrayLike = float | Sequence[float] | npt.NDArray[Any]
#: Dense array, nested list or any SciPy sparse matrix/array.
MatrixLike: TypeAlias = Any


class ObjSense(StrEnum):
    """Direction of optimisation."""

    MINIMIZE = "min"
    MAXIMIZE = "max"


class VarType(StrEnum):
    """Variable domain. Values match Gurobi's ``vtype`` characters."""

    CONTINUOUS = "C"
    INTEGER = "I"
    BINARY = "B"


class Sense(StrEnum):
    """Constraint sense. Values match Gurobi's sense characters."""

    LE = "<"
    GE = ">"
    EQ = "="


@dataclass(frozen=True)
class VarBlock:
    """A named, contiguous, possibly multi-dimensional block of variables."""

    name: str
    offset: int
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        """Number of scalar variables in the block."""
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def slice(self) -> slice:
        """Position of the block in the global variable vector."""
        return slice(self.offset, self.offset + self.size)

    def index(self) -> IntArray:
        """Global column indices, shaped like the block."""
        return np.arange(self.offset, self.offset + self.size, dtype=np.int64).reshape(self.shape)

    def values(self, x: FloatArray) -> FloatArray:
        """Extract this block's values from a global vector, reshaped to ``shape``."""
        return np.asarray(x[self.slice], dtype=np.float64).reshape(self.shape)


@dataclass(frozen=True)
class ConstrBlock:
    """A named, contiguous block of constraints sharing one sense."""

    name: str
    offset: int
    size: int
    sense: Sense

    @property
    def slice(self) -> slice:
        """Position of the block in the global constraint vector."""
        return slice(self.offset, self.offset + self.size)

    def values(self, y: FloatArray) -> FloatArray:
        """Extract this block's entries (e.g. duals) from a global vector."""
        return np.asarray(y[self.slice], dtype=np.float64)


@dataclass(frozen=True, eq=False)
class LinearModel:
    """Immutable, validated model in canonical matrix form. Build via :class:`ModelBuilder`."""

    name: str
    sense: ObjSense
    c: FloatArray
    A: sp.csr_array
    senses: npt.NDArray[np.str_]
    rhs: FloatArray
    lb: FloatArray
    ub: FloatArray
    vtypes: npt.NDArray[np.str_]
    var_blocks: Mapping[str, VarBlock]
    constr_blocks: Mapping[str, ConstrBlock]
    c0: float = 0.0
    _names: dict[str, list[str]] = field(default_factory=dict, repr=False)

    @property
    def num_vars(self) -> int:
        """Number of columns."""
        return int(self.c.shape[0])

    @property
    def num_constrs(self) -> int:
        """Number of rows."""
        return int(self.rhs.shape[0])

    @property
    def num_nonzeros(self) -> int:
        """Non-zeros in the constraint matrix."""
        return int(self.A.nnz)

    @property
    def integer_mask(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of integer or binary columns."""
        return np.asarray(self.vtypes != VarType.CONTINUOUS.value, dtype=np.bool_)

    @property
    def is_mip(self) -> bool:
        """Whether any variable is integer or binary."""
        return bool(self.integer_mask.any())

    def relaxed(self) -> LinearModel:
        """Continuous (LP) relaxation: same model with every integrality dropped.

        Binary bounds are already ``[0, 1]``, so only the variable types change.
        """
        return replace(
            self,
            name=f"{self.name}_lp",
            vtypes=np.full(self.num_vars, VarType.CONTINUOUS.value),
            _names=dict(self._names),
        )

    def with_cuts(self, cuts: Sequence[Any], name: str = "cuts") -> LinearModel:
        """A copy with extra rows appended as one constraint block ``name``.

        ``cuts`` are :class:`~linear_opt.core.result.Cut` objects (``cols``,
        ``coefs``, ``sense``, ``rhs``). Used for row generation.
        """
        if not cuts:
            return self
        if name in self.constr_blocks:
            raise ValueError(f"constraint block {name!r} already exists")
        rows = np.concatenate([np.full(len(c.cols), k) for k, c in enumerate(cuts)])
        cols = np.concatenate([c.cols for c in cuts])
        vals = np.concatenate([c.coefs for c in cuts])
        extra = sp.csr_array((vals, (rows, cols)), shape=(len(cuts), self.num_vars))
        senses = np.array([c.sense.value for c in cuts])
        blocks = dict(self.constr_blocks)
        # Mixed senses are allowed here; the block records the first one.
        blocks[name] = ConstrBlock(name, self.num_constrs, len(cuts), cuts[0].sense)
        return replace(
            self,
            A=sp.csr_array(sp.vstack([self.A, extra])),
            senses=np.concatenate([self.senses, senses]),
            rhs=np.concatenate([self.rhs, [float(c.rhs) for c in cuts]]),
            constr_blocks=MappingProxyType(blocks),
            _names={},
        )

    def var_names(self) -> list[str]:
        """Readable column names such as ``flow[3,7]`` (cached)."""
        if "vars" not in self._names:
            self._names["vars"] = _block_names(
                (b.name, b.shape) for b in sorted(self.var_blocks.values(), key=lambda b: b.offset)
            )
        return self._names["vars"]

    def constr_names(self) -> list[str]:
        """Readable row names such as ``demand[12]`` (cached)."""
        if "constrs" not in self._names:
            self._names["constrs"] = _block_names(
                (b.name, (b.size,))
                for b in sorted(self.constr_blocks.values(), key=lambda b: b.offset)
            )
        return self._names["constrs"]

    def stats(self) -> dict[str, Any]:
        """Size statistics, e.g. to check against the restricted Gurobi licence."""
        return {
            "variables": self.num_vars,
            "integer_variables": int(self.integer_mask.sum()),
            "constraints": self.num_constrs,
            "nonzeros": self.num_nonzeros,
            "is_mip": self.is_mip,
        }


def _block_names(blocks: Any) -> list[str]:
    names: list[str] = []
    for name, shape in blocks:
        names.extend(f"{name}[{','.join(map(str, idx))}]" for idx in np.ndindex(*shape))
    return names


class ModelBuilder:
    """Incrementally declare variable and constraint blocks, then :meth:`build`.

    Example:
        >>> b = ModelBuilder("toy", ObjSense.MAXIMIZE)
        >>> x = b.add_vars("x", 2, obj=[3.0, 5.0])
        >>> _ = b.add_constrs("cap", [(x, [[1, 0], [0, 2], [3, 2]])], Sense.LE, [4, 12, 18])
        >>> b.build().num_constrs
        3
    """

    def __init__(self, name: str, sense: ObjSense = ObjSense.MINIMIZE) -> None:
        self.name = name
        self.sense = sense
        self._c0 = 0.0
        self._var_blocks: dict[str, VarBlock] = {}
        self._constr_blocks: dict[str, ConstrBlock] = {}
        self._n = 0
        self._m = 0
        self._c: list[FloatArray] = []
        self._lb: list[FloatArray] = []
        self._ub: list[FloatArray] = []
        self._vt: list[npt.NDArray[np.str_]] = []
        self._rows: list[IntArray] = []
        self._cols: list[IntArray] = []
        self._vals: list[FloatArray] = []
        self._senses: list[npt.NDArray[np.str_]] = []
        self._rhs: list[FloatArray] = []

    # ------------------------------------------------------------------ variables
    def add_vars(
        self,
        name: str,
        shape: int | tuple[int, ...],
        *,
        lb: ArrayLike = 0.0,
        ub: ArrayLike = np.inf,
        vtype: VarType = VarType.CONTINUOUS,
        obj: ArrayLike = 0.0,
    ) -> VarBlock:
        """Add a block of variables.

        Args:
            name: Unique block name.
            shape: Block shape, e.g. ``(m, n)``.
            lb: Lower bounds (scalar or broadcastable to ``shape``).
            ub: Upper bounds (scalar or broadcastable to ``shape``).
            vtype: Domain of every variable in the block.
            obj: Objective coefficients (scalar or broadcastable to ``shape``).

        Returns:
            The block handle, used in constraints and to read solutions.
        """
        self._check_new_name(name, self._var_blocks)
        shape_t = (shape,) if isinstance(shape, int) else tuple(shape)
        if any(s <= 0 for s in shape_t):
            raise ValueError(f"block '{name}': shape must be positive, got {shape_t}")
        block = VarBlock(name, self._n, shape_t)

        lo = np.broadcast_to(np.asarray(lb, dtype=np.float64), shape_t).ravel()
        hi = np.broadcast_to(np.asarray(ub, dtype=np.float64), shape_t).ravel()
        if vtype is VarType.BINARY:
            lo, hi = np.maximum(lo, 0.0), np.minimum(hi, 1.0)
        if np.any(lo > hi):
            raise ValueError(f"block '{name}': lower bound exceeds upper bound")
        cost = np.broadcast_to(np.asarray(obj, dtype=np.float64), shape_t).ravel()
        if not np.all(np.isfinite(cost)):
            raise ValueError(f"block '{name}': objective coefficients must be finite")

        self._c.append(cost.copy())
        self._lb.append(lo.copy())
        self._ub.append(hi.copy())
        self._vt.append(np.full(block.size, vtype.value))
        self._var_blocks[name] = block
        self._n += block.size
        return block

    # ---------------------------------------------------------------- constraints
    def add_constrs(
        self,
        name: str,
        terms: Sequence[tuple[VarBlock, MatrixLike]],
        sense: Sense,
        rhs: ArrayLike,
    ) -> ConstrBlock:
        r"""Add the block of constraints :math:`\sum_k M_k x_{B_k} \;\text{sense}\; b`.

        Args:
            name: Unique block name.
            terms: Pairs ``(block, M)`` where ``M`` has shape ``(rows, block.size)``
                and acts on the block's variables in row-major (C) order.
            sense: Common sense of all rows.
            rhs: Right-hand side, scalar or of length ``rows``.

        Returns:
            The constraint block handle (used to read duals).
        """
        self._check_new_name(name, self._constr_blocks)
        if not terms:
            raise ValueError(f"constraint block '{name}' has no terms")
        n_rows: int | None = None
        for block, matrix in terms:
            if self._var_blocks.get(block.name) is not block:
                raise ValueError(
                    f"constraint '{name}': block '{block.name}' is not from this model"
                )
            coo = sp.coo_array(matrix)
            rows, cols = coo.shape
            if cols != block.size:
                raise ValueError(
                    f"constraint '{name}': matrix for '{block.name}' has {cols} columns, "
                    f"block has {block.size} variables"
                )
            if n_rows is not None and rows != n_rows:
                raise ValueError(f"constraint '{name}': terms disagree on the number of rows")
            n_rows = rows
            self._rows.append(coo.row.astype(np.int64) + self._m)
            self._cols.append(coo.col.astype(np.int64) + block.offset)
            self._vals.append(coo.data.astype(np.float64))
        assert n_rows is not None
        b = np.broadcast_to(np.asarray(rhs, dtype=np.float64), (n_rows,)).copy()
        if np.any(np.isnan(b)):
            raise ValueError(f"constraint '{name}': right-hand side contains NaN")

        cblock = ConstrBlock(name, self._m, n_rows, sense)
        self._senses.append(np.full(n_rows, sense.value))
        self._rhs.append(b)
        self._constr_blocks[name] = cblock
        self._m += n_rows
        return cblock

    def set_objective_constant(self, value: float) -> None:
        """Set the constant :math:`c_0` in the objective."""
        self._c0 = float(value)

    # ---------------------------------------------------------------------- build
    def build(self) -> LinearModel:
        """Assemble and freeze the model."""
        if self._n == 0:
            raise ValueError("model has no variables")
        cat = np.concatenate
        rows = cat(self._rows) if self._rows else np.empty(0, np.int64)
        cols = cat(self._cols) if self._cols else np.empty(0, np.int64)
        vals = cat(self._vals) if self._vals else np.empty(0, np.float64)
        matrix = sp.csr_array(sp.coo_array((vals, (rows, cols)), shape=(self._m, self._n)))
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return LinearModel(
            name=self.name,
            sense=self.sense,
            c=cat(self._c),
            A=matrix,
            senses=cat(self._senses) if self._senses else np.empty(0, "<U1"),
            rhs=cat(self._rhs) if self._rhs else np.empty(0, np.float64),
            lb=cat(self._lb),
            ub=cat(self._ub),
            vtypes=cat(self._vt),
            var_blocks=MappingProxyType(dict(self._var_blocks)),
            constr_blocks=MappingProxyType(dict(self._constr_blocks)),
            c0=self._c0,
        )

    @staticmethod
    def _check_new_name(name: str, existing: Mapping[str, object]) -> None:
        if not name.isidentifier():
            raise ValueError(f"block name must be a valid identifier, got {name!r}")
        if name in existing:
            raise ValueError(f"duplicate block name {name!r}")


# --------------------------------------------------------------------- helpers
def sum_over(shape: tuple[int, int], axis: int) -> sp.csr_array:
    r"""Matrix that sums a row-major ``(m, n)`` block along ``axis``.

    ``axis=1`` gives :math:`I_m \otimes \mathbf{1}_n^\top` (one row per ``i``,
    summing over ``j``); ``axis=0`` gives :math:`\mathbf{1}_m^\top \otimes I_n`.
    """
    m, n = shape
    if axis == 1:
        return sp.csr_array(sp.kron(sp.eye_array(m), np.ones((1, n))))
    if axis == 0:
        return sp.csr_array(sp.kron(np.ones((1, m)), sp.eye_array(n)))
    raise ValueError("axis must be 0 or 1")
