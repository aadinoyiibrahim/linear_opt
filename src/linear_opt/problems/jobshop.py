r"""Job-shop scheduling: minimise the makespan :math:`C_{\max}` (MILP).

Every job visits machines in a fixed order; operation :math:`o` needs machine
:math:`\mu_o` for :math:`p_o` time units; a machine processes one operation at a
time and operations are not interrupted.

* **Disjunctive** (Manne 1960): start times :math:`s_o`, job precedences, and one
  binary :math:`y_{ab}` per pair of operations on the same machine choosing
  which goes first, linked through big-M constraints

  .. math::

      s_a + p_a \le s_b + M_{ab}(1 - y_{ab}), \qquad
      s_b + p_b \le s_a + M_{ba}\, y_{ab}.

  :math:`M` is derived from a horizon :math:`H` (an upper bound on the optimal
  makespan): ``big_m = "tight"`` takes :math:`H` from a Giffler-Thompson
  schedule, ``"naive"`` uses :math:`H = \sum_o p_o`. Same integer solutions,
  very different LP relaxations - the classic big-M lesson.
* **Time-indexed** (Bowman 1959): :math:`x_{o,t} = 1` if :math:`o` starts at
  integer time :math:`t`. Machine capacity is a clique constraint per machine and
  period. The LP relaxation is much stronger, but the model grows with the
  horizon, so it is practical only for short integer processing times. Start
  windows are trimmed with heads and tails: :math:`t \in [r_o, H - q_o - p_o]`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict

from linear_opt.core.config import ProblemKind
from linear_opt.core.ir import FloatArray, IntArray, ModelBuilder, Sense, VarBlock, VarType
from linear_opt.core.model import OptimizationModel
from linear_opt.core.result import ProgressPoint, RawSolution
from linear_opt.data.jsplib import JobShopData, Reference

TOL = 1e-6


@dataclass(frozen=True, eq=False)
class JobShopInstance:
    """Immutable job-shop instance; operation ``o = j * n_ops + k`` is step ``k`` of job ``j``."""

    name: str
    machines: npt.NDArray[np.int64]  # (n_jobs, n_ops)
    durations: FloatArray  # (n_jobs, n_ops)
    reference: float | None = None
    lower_bound_ref: float | None = None
    upper_bound_ref: float | None = None

    def __post_init__(self) -> None:
        if self.machines.ndim != 2 or self.machines.shape != self.durations.shape:
            raise ValueError("machines and durations must be (n_jobs, n_ops) arrays of equal shape")
        if self.machines.size == 0:
            raise ValueError("empty instance")
        if self.machines.min() < 0:
            raise ValueError("machine indices must be non-negative")
        if np.any(self.durations < 0) or not np.all(np.isfinite(self.durations)):
            raise ValueError("durations must be finite and non-negative")

    # ---------------------------------------------------------------- shape
    @property
    def n_jobs(self) -> int:
        """Number of jobs."""
        return int(self.machines.shape[0])

    @property
    def n_ops(self) -> int:
        """Operations per job."""
        return int(self.machines.shape[1])

    @property
    def n_machines(self) -> int:
        """Number of machines."""
        return int(self.machines.max()) + 1

    @property
    def mu(self) -> IntArray:
        """Machine of every operation (flattened)."""
        return np.asarray(self.machines.ravel(), dtype=np.int64)

    @property
    def p(self) -> FloatArray:
        """Duration of every operation (flattened)."""
        return np.asarray(self.durations.ravel(), dtype=np.float64)

    @property
    def heads(self) -> FloatArray:
        """Earliest start ``r_o``: work of the job's earlier operations."""
        return np.asarray(np.cumsum(self.durations, axis=1) - self.durations).ravel()

    @property
    def tails(self) -> FloatArray:
        """Work of the job's later operations ``q_o`` (excluding ``o``)."""
        rev = np.cumsum(self.durations[:, ::-1], axis=1)[:, ::-1]
        return np.asarray(rev - self.durations).ravel()

    @property
    def lower_bound(self) -> float:
        """Trivial bound: the longest job or the most loaded machine."""
        machine_load = np.bincount(self.mu, weights=self.p, minlength=self.n_machines)
        return float(max(self.durations.sum(axis=1).max(), machine_load.max()))

    def machine_pairs(self) -> list[tuple[int, int]]:
        """All pairs ``(a, b)``, ``a < b``, of operations sharing a machine."""
        pairs: list[tuple[int, int]] = []
        for m in range(self.n_machines):
            ops = np.flatnonzero(self.mu == m)
            pairs.extend((int(a), int(b)) for i, a in enumerate(ops) for b in ops[i + 1 :])
        return pairs

    def label(self, o: int) -> str:
        """Human-readable operation name, e.g. ``J3·2`` (job 3, step 2), 1-based."""
        return f"J{o // self.n_ops + 1}·{o % self.n_ops + 1}"

    # ------------------------------------------------------------ factories
    @classmethod
    def from_data(cls, data: JobShopData, ref: Reference | None = None) -> JobShopInstance:
        """Wrap a parsed benchmark instance."""
        return cls(
            data.name,
            data.machines,
            data.durations,
            reference=ref.optimum if ref else None,
            lower_bound_ref=ref.lower if ref else None,
            upper_bound_ref=ref.upper if ref else None,
        )

    @classmethod
    def random(
        cls, n_jobs: int, n_machines: int, *, seed: int = 0, max_duration: int = 10
    ) -> JobShopInstance:
        """Taillard-style: each job visits every machine once in random order."""
        rng = np.random.default_rng(seed)
        machines = np.array([rng.permutation(n_machines) for _ in range(n_jobs)], dtype=np.int64)
        durations = rng.integers(1, max_duration + 1, (n_jobs, n_machines)).astype(np.float64)
        return cls(f"random-{n_jobs}x{n_machines}", machines, durations)


# --------------------------------------------------------------- heuristics
def giffler_thompson(inst: JobShopInstance, rule: Literal["mwkr", "spt"] = "mwkr") -> FloatArray:
    """Active schedule (Giffler & Thompson 1960) with a priority rule.

    Repeatedly take the schedulable operation with the earliest possible
    completion ``C*``, consider all operations on its machine that could start
    before ``C*`` (the conflict set) and schedule the one with the highest
    priority: most work remaining (``mwkr``) or shortest processing time (``spt``).

    Returns:
        Start times, shape ``(n_jobs, n_ops)``.
    """
    n, k = inst.n_jobs, inst.n_ops
    nxt = np.zeros(n, dtype=np.int64)
    job_ready = np.zeros(n)
    machine_ready = np.zeros(inst.n_machines)
    remaining = inst.durations.sum(axis=1)
    starts = np.zeros((n, k))
    for _ in range(n * k):
        jobs = np.flatnonzero(nxt < k)
        mach = inst.machines[jobs, nxt[jobs]]
        dur = inst.durations[jobs, nxt[jobs]]
        est = np.maximum(job_ready[jobs], machine_ready[mach])
        finish = est + dur
        star = int(np.argmin(finish))
        m_star, c_star = mach[star], finish[star]
        conflict = np.flatnonzero((mach == m_star) & (est < c_star - TOL))
        score = remaining[jobs[conflict]] if rule == "mwkr" else -dur[conflict]
        pick = conflict[int(np.argmax(score))]
        j = int(jobs[pick])
        s = float(est[pick])
        starts[j, nxt[j]] = s
        end = s + float(dur[pick])
        job_ready[j] = machine_ready[m_star] = end
        remaining[j] -= dur[pick]
        nxt[j] += 1
    return starts


def best_heuristic(inst: JobShopInstance) -> FloatArray:
    """Best of the Giffler-Thompson schedules for both priority rules."""
    candidates = [giffler_thompson(inst, rule) for rule in ("mwkr", "spt")]
    return min(candidates, key=lambda s: float((s + inst.durations).max()))


def left_shift(inst: JobShopInstance, starts: FloatArray) -> FloatArray:
    """Semi-active schedule with the same machine sequences.

    Each operation starts as soon as its job predecessor and machine predecessor
    have finished. Processing operations in order of their original start time
    respects both precedence types, so one pass suffices; the makespan cannot
    increase. MIP solutions are often not left-justified (non-critical
    operations may float), which would hide the critical path.
    """
    s, p, mu = starts.ravel(), inst.p, inst.mu
    out = np.zeros_like(s)
    job_ready = np.zeros(inst.n_jobs)
    machine_ready = np.zeros(inst.n_machines)
    for o in np.lexsort((np.arange(len(s)), s)):
        j, m = o // inst.n_ops, mu[o]
        out[o] = max(job_ready[j], machine_ready[m])
        job_ready[j] = machine_ready[m] = out[o] + p[o]
    return out.reshape(starts.shape)


def critical_path(inst: JobShopInstance, starts: FloatArray) -> list[int]:
    """One critical path (operations without slack), from first to last.

    Walk back from an operation that ends at the makespan: its predecessor is the
    job predecessor or the machine predecessor that ends exactly at its start.
    """
    s, p, mu = starts.ravel(), inst.p, inst.mu
    end = s + p
    cur = int(np.argmax(end))
    path = [cur]
    while s[cur] > TOL:
        cands = []
        if cur % inst.n_ops > 0:
            cands.append(cur - 1)
        cands += [int(o) for o in np.flatnonzero((mu == mu[cur]) & (np.arange(len(s)) != cur))]
        preds = [o for o in cands if abs(end[o] - s[cur]) < TOL and s[o] < s[cur] + TOL]
        if not preds:
            break  # idle time before cur: the path starts here
        cur = preds[0]
        path.append(cur)
    return path[::-1]


# -------------------------------------------------------------------- model
class JobShopOptions(BaseModel):
    """Options in the ``[model]`` table of a job-shop config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    formulation: Literal["disjunctive", "time_indexed"] = "disjunctive"
    big_m: Literal["tight", "naive"] = "tight"
    warm_start: Literal["giffler_thompson", "none"] = "giffler_thompson"


@dataclass(frozen=True, eq=False)
class JobShopSolution:
    """Start times, makespan and a critical path."""

    starts: FloatArray  # (n_jobs, n_ops)
    makespan: float
    objective: float
    critical: tuple[int, ...]
    progress: tuple[ProgressPoint, ...] = ()
    #: Tolerance-level violations in the raw solver output that the left shift
    #: repaired (big-M models with large M are prone to these).
    repaired_violations: int = 0


class JobShopModel(OptimizationModel[JobShopInstance, JobShopSolution, JobShopOptions]):
    """Job-shop scheduling on a :class:`JobShopInstance`."""

    kind: ClassVar[ProblemKind] = ProblemKind.JOB_SHOP
    options_type: ClassVar[type[BaseModel]] = JobShopOptions

    _start: VarBlock
    _cmax: VarBlock
    _order: VarBlock | None
    _x_op: IntArray
    _x_t: FloatArray

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._heuristic: FloatArray | None = None

    # -------------------------------------------------------------- horizon
    def heuristic_starts(self) -> FloatArray:
        """Best Giffler-Thompson schedule (cached)."""
        if self._heuristic is None:
            self._heuristic = best_heuristic(self.data)
        return self._heuristic

    def horizon(self) -> float:
        """Upper bound ``H`` on the optimal makespan used for bounds and big-M."""
        inst = self.data
        if self.options.formulation == "disjunctive" and self.options.big_m == "naive":
            return float(inst.p.sum())
        return float((self.heuristic_starts() + inst.durations).max())

    # ----------------------------------------------------------------- build
    def _build(self, builder: ModelBuilder) -> None:
        if self.options.formulation == "disjunctive":
            self._build_disjunctive(builder)
        else:
            self._build_time_indexed(builder)

    def _precedence_rows(self) -> tuple[IntArray, IntArray]:
        n, k = self.data.n_jobs, self.data.n_ops
        before = np.array([j * k + s for j in range(n) for s in range(k - 1)], dtype=np.int64)
        return before, before + 1

    def _build_disjunctive(self, builder: ModelBuilder) -> None:
        inst, horizon = self.data, self.horizon()
        p, r, q = inst.p, inst.heads, inst.tails
        n_op = len(p)
        lb, ub = r, horizon - q - p
        shape = (inst.n_jobs, inst.n_ops)
        self._start = builder.add_vars("start", shape, lb=lb.reshape(shape), ub=ub.reshape(shape))
        self._cmax = builder.add_vars("makespan", 1, ub=horizon, obj=1.0)

        before, after = self._precedence_rows()
        rows = np.arange(len(before))
        prec = sp.csr_array(
            (
                np.r_[np.ones(len(rows)), -np.ones(len(rows))],
                (np.r_[rows, rows], np.r_[after, before]),
            ),
            shape=(len(rows), n_op),
        )
        builder.add_constrs("precedence", [(self._start, prec)], Sense.GE, p[before])

        last = np.arange(inst.n_jobs) * inst.n_ops + inst.n_ops - 1
        jrows = np.arange(inst.n_jobs)
        pick_last = sp.csr_array((-np.ones(inst.n_jobs), (jrows, last)), shape=(inst.n_jobs, n_op))
        builder.add_constrs(
            "finish",
            [(self._cmax, np.ones((inst.n_jobs, 1))), (self._start, pick_last)],
            Sense.GE,
            p[last],
        )

        pairs = np.array(inst.machine_pairs(), dtype=np.int64).reshape(-1, 2)
        self._order = None
        if len(pairs) == 0:
            return
        a, b = pairs[:, 0], pairs[:, 1]
        self._order = builder.add_vars("order", len(pairs), vtype=VarType.BINARY)
        m_ab = ub[a] + p[a] - lb[b]  # tightest valid M when b goes first (y = 0)
        m_ba = ub[b] + p[b] - lb[a]  # ... when a goes first (y = 1)
        k = np.arange(len(pairs))
        # y = 1 (a before b):  s_a - s_b + M_ab * y <= M_ab - p_a
        s1 = sp.csr_array(
            (np.r_[np.ones(len(k)), -np.ones(len(k))], (np.r_[k, k], np.r_[a, b])),
            shape=(len(k), n_op),
        )
        builder.add_constrs(
            "a_before_b",
            [(self._start, s1), (self._order, sp.diags_array(m_ab))],
            Sense.LE,
            m_ab - p[a],
        )
        # y = 0 (b before a):  s_b - s_a - M_ba * y <= -p_b
        builder.add_constrs(
            "b_before_a",
            [(self._start, -s1), (self._order, sp.diags_array(-m_ba))],
            Sense.LE,
            -p[b],
        )

    def _build_time_indexed(self, builder: ModelBuilder) -> None:
        inst = self.data
        if not np.allclose(inst.p, np.round(inst.p)):
            raise ValueError("time-indexed formulation needs integer processing times")
        horizon = round(self.horizon())
        p = np.round(inst.p).astype(np.int64)
        r = np.round(inst.heads).astype(np.int64)
        q = np.round(inst.tails).astype(np.int64)
        first, last = r, horizon - q - p
        if np.any(last < first):
            raise ValueError("horizon too short for some operation")
        op = np.concatenate(
            [np.full(int(e - s + 1), o) for o, (s, e) in enumerate(zip(first, last, strict=True))]
        )
        t = np.concatenate([np.arange(s, e + 1) for s, e in zip(first, last, strict=True)])
        self._x_op, self._x_t = op.astype(np.int64), t.astype(np.float64)
        n_x, n_op = len(op), len(p)
        self._start = builder.add_vars("start_at", n_x, vtype=VarType.BINARY)
        self._cmax = builder.add_vars("makespan", 1, ub=horizon, obj=1.0)
        self._order = None
        cols = np.arange(n_x)

        assign = sp.csr_array((np.ones(n_x), (op, cols)), shape=(n_op, n_x))
        builder.add_constrs("assign", [(self._start, assign)], Sense.EQ, 1.0)

        # capacity: x_{o,tau} occupies machine mu_o during tau .. tau + p_o - 1
        busy_rows, busy_cols = [], []
        for c in range(n_x):
            o = op[c]
            span = np.arange(t[c], t[c] + p[o])
            busy_rows.append(inst.mu[o] * horizon + span)
            busy_cols.append(np.full(len(span), c))
        rows = np.concatenate(busy_rows) if busy_rows else np.empty(0, np.int64)
        used, rows = np.unique(rows, return_inverse=True)
        cap = sp.csr_array(
            (np.ones(len(rows)), (rows, np.concatenate(busy_cols))), shape=(len(used), n_x)
        )
        builder.add_constrs("capacity", [(self._start, cap)], Sense.LE, 1.0)

        # Strong (disaggregated) precedence: o+1 can have started by time tt only
        # if o started by tt - p_o:  sum_{tau<=tt} x_{o+1,tau} <= sum_{tau<=tt-p_o} x_{o,tau}.
        # The aggregated form sum t x_{o+1,t} >= sum t x_{o,t} + p_o is much weaker.
        before, after = self._precedence_rows()
        start_of = sp.csr_array((t.astype(np.float64), (op, cols)), shape=(n_op, n_x))
        p_rows: list[int] = []
        p_cols: list[int] = []
        p_vals: list[float] = []
        row = 0
        for o, nxt in zip(before, after, strict=True):
            mine = np.flatnonzero(op == o)
            theirs = np.flatnonzero(op == nxt)
            for tt in t[theirs][:-1]:  # the last window point is implied by "assign"
                lhs = theirs[t[theirs] <= tt]
                rhs = mine[t[mine] <= tt - p[o]]
                p_rows += [row] * (len(lhs) + len(rhs))
                p_cols += [*lhs.tolist(), *rhs.tolist()]
                p_vals += [1.0] * len(lhs) + [-1.0] * len(rhs)
                row += 1
        prec = sp.csr_array((p_vals, (p_rows, p_cols)), shape=(row, n_x))
        builder.add_constrs("precedence", [(self._start, prec)], Sense.LE, 0.0)
        last_ops = np.arange(inst.n_jobs) * inst.n_ops + inst.n_ops - 1
        builder.add_constrs(
            "finish",
            [(self._cmax, np.ones((inst.n_jobs, 1))), (self._start, -start_of[last_ops])],
            Sense.GE,
            p[last_ops].astype(float),
        )

    # ------------------------------------------------------------ warm start
    def warm_start(self) -> FloatArray | None:
        """MIP start from the best Giffler-Thompson schedule."""
        if self.options.warm_start == "none":
            return None
        inst = self.data
        starts = self.heuristic_starts()
        vec = np.zeros(self.build().num_vars)
        cmax = float((starts + inst.durations).max())
        if cmax > self.horizon() + TOL:
            return None
        vec[self._cmax.offset] = cmax
        s = starts.ravel()
        if self.options.formulation == "disjunctive":
            vec[self._start.slice] = s
            if self._order is not None:
                pairs = np.array(inst.machine_pairs(), dtype=np.int64)
                vec[self._order.slice] = (s[pairs[:, 0]] < s[pairs[:, 1]]).astype(float)
        else:
            hit = np.flatnonzero(np.isclose(self._x_t, s[self._x_op]))
            vec[self._start.offset + hit] = 1.0
        return vec

    def reference_objective(self) -> float | None:
        """Published optimal makespan (JSPLIB), if known."""
        return self.data.reference

    @classmethod
    def variants(cls) -> dict[str, dict[str, Any]]:
        """Formulations compared by ``linopt variants``."""
        return {
            "disjunctive (tight M)": {"formulation": "disjunctive", "big_m": "tight"},
            "disjunctive (naive M)": {"formulation": "disjunctive", "big_m": "naive"},
            "time-indexed": {"formulation": "time_indexed"},
        }

    # ------------------------------------------------------------- solution
    def _extract(self, raw: RawSolution) -> JobShopSolution:
        assert raw.x is not None
        assert raw.objective is not None
        inst = self.data
        if self.options.formulation == "disjunctive":
            starts = self._start.values(raw.x)
        else:
            chosen = raw.x[self._start.slice] > 0.5
            flat = np.zeros(inst.n_jobs * inst.n_ops)
            flat[self._x_op[chosen]] = self._x_t[chosen]
            starts = flat.reshape(inst.n_jobs, inst.n_ops)
        starts = np.where(np.abs(starts - np.round(starts)) < 1e-6, np.round(starts), starts)
        raw_issues = schedule_violations(inst, starts)
        starts = left_shift(inst, starts)  # canonical semi-active form; repairs tiny overlaps
        makespan = float((starts + inst.durations).max())
        return JobShopSolution(
            starts=starts,
            makespan=makespan,
            objective=raw.objective,
            critical=tuple(critical_path(inst, starts)),
            progress=raw.progress,
            repaired_violations=len(raw_issues),
        )

    def validate(self, solution: JobShopSolution) -> list[str]:
        """Job order, machine overlaps and makespan - recomputed from the data."""
        return schedule_violations(self.data, solution.starts, solution.objective)


def schedule_violations(
    inst: JobShopInstance, starts: FloatArray, objective: float | None = None
) -> list[str]:
    """Constraint violations of a schedule (empty if feasible)."""
    issues: list[str] = []
    s, p, mu = starts.ravel(), inst.p, inst.mu
    if np.any(s < -TOL):
        issues.append("negative start time")
    for o in range(len(s)):
        if o % inst.n_ops and s[o] < s[o - 1] + p[o - 1] - 1e-5:
            issues.append(f"{inst.label(o)} starts before {inst.label(o - 1)} ends")
    for m in range(inst.n_machines):
        ops = np.flatnonzero(mu == m)
        order = ops[np.argsort(s[ops])]
        for a, b in pairwise(order):
            if s[b] < s[a] + p[a] - 1e-5:
                issues.append(
                    f"machine {m + 1}: {inst.label(int(a))} overlaps {inst.label(int(b))}"
                )
    makespan = float((s + p).max())
    if objective is not None and not np.isclose(makespan, objective, rtol=1e-6, atol=1e-5):
        issues.append(f"objective {objective:.6g} != makespan {makespan:.6g}")
    return issues


def machine_sequences(inst: JobShopInstance, starts: FloatArray) -> list[Sequence[int]]:
    """Operations on each machine in processing order."""
    s = starts.ravel()
    return [
        [int(o) for o in sorted(np.flatnonzero(inst.mu == m), key=lambda o: s[o])]
        for m in range(inst.n_machines)
    ]
