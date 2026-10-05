"""Gurobi backend built on the gurobipy matrix API.

Showcases:

* ``addMVar`` / ``addMConstr`` - one call per block, no Python loops over variables;
* a callback that (a) adds **lazy constraints** at integer-feasible nodes via
  ``cbLazy`` and (b) records the **incumbent / bound trajectory**;
* **MIP starts** through ``MVar.Start``;
* **IIS** computation for infeasible models;
* a clean mapping of the size-restricted pip licence error to
  :class:`LicenseLimitError`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from linear_opt.core.config import BackendName, SolverSettings
from linear_opt.core.ir import FloatArray, LinearModel, ObjSense
from linear_opt.core.result import LazyOracle, ProgressRecorder, RawSolution, SolveStatus
from linear_opt.solvers.base import (
    Capability,
    LicenseLimitError,
    SolverBackend,
    solve_by_row_generation,
)

_STATUS: dict[int, SolveStatus] = {
    GRB.OPTIMAL: SolveStatus.OPTIMAL,
    GRB.INFEASIBLE: SolveStatus.INFEASIBLE,
    GRB.INF_OR_UNBD: SolveStatus.INF_OR_UNBD,
    GRB.UNBOUNDED: SolveStatus.UNBOUNDED,
    GRB.TIME_LIMIT: SolveStatus.TIME_LIMIT,
    GRB.INTERRUPTED: SolveStatus.INTERRUPTED,
    GRB.NUMERIC: SolveStatus.NUMERIC,
}
_SIZE_LIMIT_ERRNO = 10010  # "Model too large for size-limited license"


def _finite(a: FloatArray) -> FloatArray:
    """Map +-inf to Gurobi's +-GRB.INFINITY."""
    return np.clip(a, -GRB.INFINITY, GRB.INFINITY)


class _Callback:
    """Gurobi callback: lazy constraints + progress recording."""

    #: Separation rounds allowed at the root node for user cuts.
    MAX_ROOT_ROUNDS = 100

    def __init__(
        self, variables: list[gp.Var], lazy: LazyOracle | None, user_cuts: LazyOracle | None = None
    ) -> None:
        self.variables = variables
        self.lazy = lazy
        self.user_cuts = user_cuts
        self.progress = ProgressRecorder(infinity=GRB.INFINITY)
        self.lazy_cuts = 0
        self.user_cut_count = 0
        self.root_rounds = 0

    def _add(self, model: gp.Model, cuts: Any, lazy: bool) -> int:
        added = 0
        for cut in cuts:
            expr = gp.LinExpr(cut.coefs.tolist(), [self.variables[i] for i in cut.cols])
            if lazy:
                model.cbLazy(expr, cut.sense.value, cut.rhs)
            else:
                model.cbCut(expr, cut.sense.value, cut.rhs)
            added += 1
        return added

    def __call__(self, model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIP:
            self.progress.record(
                model.cbGet(GRB.Callback.RUNTIME),
                model.cbGet(GRB.Callback.MIP_OBJBST),
                model.cbGet(GRB.Callback.MIP_OBJBND),
            )
        elif where == GRB.Callback.MIPSOL:
            self.progress.record(
                model.cbGet(GRB.Callback.RUNTIME),
                model.cbGet(GRB.Callback.MIPSOL_OBJBST),
                model.cbGet(GRB.Callback.MIPSOL_OBJBND),
            )
            if self.lazy is not None:
                x = np.asarray(model.cbGetSolution(self.variables), dtype=np.float64)
                self.lazy_cuts += self._add(model, self.lazy(x), lazy=True)
        elif (
            where == GRB.Callback.MIPNODE
            and self.user_cuts is not None
            and self.root_rounds < self.MAX_ROOT_ROUNDS
            and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL
            and model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0
        ):
            self.root_rounds += 1
            x = np.asarray(model.cbGetNodeRel(self.variables), dtype=np.float64)
            self.user_cut_count += self._add(model, self.user_cuts(x), lazy=False)


class GurobiBackend(SolverBackend):
    """Gurobi via gurobipy (works with the bundled restricted licence)."""

    name: ClassVar[BackendName] = BackendName.GUROBI
    capabilities: ClassVar[frozenset[Capability]] = frozenset(Capability)

    def version(self) -> str:  # noqa: D102
        return ".".join(map(str, gp.gurobi.version()))

    # --- helpers
    @staticmethod
    @contextmanager
    def _environment(settings: SolverSettings) -> Iterator[gp.Env]:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", int(settings.verbose))
        try:
            env.start()
            yield env
        finally:
            env.dispose()

    @staticmethod
    def _populate(m: gp.Model, model: LinearModel) -> tuple[gp.MVar, list[gp.MConstr]]:
        x = m.addMVar(
            model.num_vars,
            lb=_finite(model.lb),
            ub=_finite(model.ub),
            vtype=model.vtypes,
            obj=model.c,
            name=model.var_names(),
        )
        m.ModelSense = GRB.MINIMIZE if model.sense is ObjSense.MINIMIZE else GRB.MAXIMIZE
        m.ObjCon = model.c0
        constrs = [
            m.addMConstr(
                model.A[block.slice],
                x,
                model.senses[block.slice],
                model.rhs[block.slice],
                name=block.name,
            )
            for block in sorted(model.constr_blocks.values(), key=lambda b: b.offset)
        ]
        return x, constrs

    @staticmethod
    def _apply(m: gp.Model, settings: SolverSettings, lazy: bool, user_cuts: bool = False) -> None:
        p = m.Params
        p.TimeLimit = settings.time_limit
        p.MIPGap = settings.mip_gap
        p.Seed = settings.seed
        if settings.threads:
            p.Threads = settings.threads
        if lazy:
            p.LazyConstraints = 1
        if user_cuts:
            p.PreCrush = 1  # cuts refer to the original (not presolved) model
        for name, value in settings.gurobi.items():
            m.setParam(name, value)

    @contextmanager
    def _translate_errors(self) -> Iterator[None]:
        try:
            yield
        except gp.GurobiError as exc:
            if exc.errno == _SIZE_LIMIT_ERRNO:
                raise LicenseLimitError(
                    "model exceeds the restricted Gurobi licence (about 2000 variables / "
                    "constraints). Use a full licence (GRB_LICENSE_FILE) or `--backend highs`."
                ) from exc
            raise

    # --- solve
    def solve(  # noqa: D102
        self,
        model: LinearModel,
        settings: SolverSettings,
        *,
        lazy: LazyOracle | None = None,
        warm_start: FloatArray | None = None,
        user_cuts: LazyOracle | None = None,
    ) -> RawSolution:
        if lazy is not None and not model.is_mip:
            # No branch-and-bound callback for LPs: separate by row generation.
            return solve_by_row_generation(lambda lm, s: self.solve(lm, s), model, settings, lazy)
        user_cuts = user_cuts if model.is_mip else None
        t0 = time.perf_counter()
        with (
            self._translate_errors(),
            self._environment(settings) as env,
            gp.Model(model.name, env=env) as m,
        ):
            x, constrs = self._populate(m, model)
            self._apply(m, settings, lazy is not None, user_cuts is not None)
            if warm_start is not None:
                x.setAttr("Start", np.where(np.isnan(warm_start), GRB.UNDEFINED, warm_start))
            if settings.export is not None:
                m.write(str(settings.export))
            callback = _Callback(x.tolist(), lazy, user_cuts)
            m.optimize(callback)
            return self._collect(m, model, x, constrs, callback, time.perf_counter() - t0)

    def _collect(
        self,
        m: gp.Model,
        model: LinearModel,
        x: gp.MVar,
        constrs: list[gp.MConstr],
        callback: _Callback,
        wall: float,
    ) -> RawSolution:
        status = _STATUS.get(m.Status, SolveStatus.OTHER)
        has_solution = m.SolCount > 0
        objective = float(m.ObjVal) if has_solution else None
        bound: float | None = None
        duals = reduced = None
        if model.is_mip:
            bound = float(m.ObjBound) if has_solution or status is SolveStatus.TIME_LIMIT else None
        elif status is SolveStatus.OPTIMAL:
            bound = objective
            duals = (
                np.concatenate([np.atleast_1d(c.Pi) for c in constrs])
                if constrs
                else np.empty(0, np.float64)
            )
            reduced = np.asarray(x.RC, dtype=np.float64)
        if model.is_mip:
            # Close the trajectory with the final state; presolve may solve a model
            # before the callback ever sees an incumbent.
            inf = GRB.INFINITY
            callback.progress.record(
                float(m.Runtime),
                inf if objective is None else objective,
                inf if bound is None else bound,
            )
        extra: dict[str, Any] = {"wall_time": wall}
        if callback.lazy is not None:
            extra["lazy_cuts"] = callback.lazy_cuts
        if callback.user_cuts is not None:
            extra["user_cuts"] = callback.user_cut_count
        return RawSolution(
            status=status,
            backend=self.name.value,
            backend_version=self.version(),
            objective=objective,
            bound=bound,
            x=np.asarray(x.X, dtype=np.float64) if has_solution else None,
            duals=duals,
            reduced_costs=reduced,
            runtime=float(m.Runtime),
            iterations=int(m.IterCount),
            nodes=int(m.NodeCount) if model.is_mip else 0,
            progress=tuple(callback.progress.points),
            extra=extra,
        )

    def write_model(self, model: LinearModel, path: str) -> None:
        """Write the model with Gurobi's writer (``.lp``, ``.mps``, ...)."""
        with (
            self._translate_errors(),
            self._environment(SolverSettings()) as env,
            gp.Model(model.name, env=env) as m,
        ):
            self._populate(m, model)
            m.write(path)

    # ------- IIS
    def compute_iis(self, model: LinearModel, settings: SolverSettings) -> list[str]:
        """Return names of constraints and bounds forming an IIS (empty if feasible)."""
        with (
            self._translate_errors(),
            self._environment(settings) as env,
            gp.Model(model.name, env=env) as m,
        ):
            self._populate(m, model)
            self._apply(m, settings, lazy=False)
            m.Params.DualReductions = 0  # distinguish infeasible from unbounded
            m.optimize()
            if m.Status != GRB.INFEASIBLE:
                return []
            m.computeIIS()
            names = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
            for v in m.getVars():
                if v.IISLB:
                    names.append(f"{v.VarName} >= {v.LB:g}")
                if v.IISUB:
                    names.append(f"{v.VarName} <= {v.UB:g}")
            return names
