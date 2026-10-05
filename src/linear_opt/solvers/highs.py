"""HiGHS backend (open source, MIT licence) via ``highspy``.

The model is passed in one shot as a column-wise ``HighsLp``. Lazy constraints
are handled by **row generation**: solve, ask the oracle for violated cuts,
append them with ``addRow`` and re-solve until no cut is violated. This gives
the same optimum as in-tree lazy constraints, at the price of re-solving.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import highspy
import numpy as np

from linear_opt.core.config import BackendName, SolverSettings
from linear_opt.core.ir import FloatArray, LinearModel, ObjSense, Sense
from linear_opt.core.result import Cut, LazyOracle, ProgressRecorder, RawSolution, SolveStatus
from linear_opt.solvers.base import Capability, SolverBackend

_MS = highspy.HighsModelStatus
_STATUS: dict[Any, SolveStatus] = {
    _MS.kOptimal: SolveStatus.OPTIMAL,
    _MS.kInfeasible: SolveStatus.INFEASIBLE,
    _MS.kUnbounded: SolveStatus.UNBOUNDED,
    _MS.kUnboundedOrInfeasible: SolveStatus.INF_OR_UNBD,
    _MS.kTimeLimit: SolveStatus.TIME_LIMIT,
    _MS.kInterrupt: SolveStatus.INTERRUPTED,
}
_FEASIBLE = 2  # HighsSolutionStatus kSolutionStatusFeasible


def _row_bounds(senses: np.ndarray[Any, Any], rhs: FloatArray) -> tuple[FloatArray, FloatArray]:
    lower = np.where(senses == Sense.LE.value, -np.inf, rhs)
    upper = np.where(senses == Sense.GE.value, np.inf, rhs)
    return lower.astype(np.float64), upper.astype(np.float64)


_LP_SAFE = str.maketrans({"[": "_", "]": "", ",": "_"})


def _lp_safe(names: list[str]) -> list[str]:
    """HiGHS discards names with brackets when writing LP files: ``flow[3,7]`` -> ``flow_3_7``."""
    return [n.translate(_LP_SAFE) for n in names]


def to_highs_lp(model: LinearModel) -> highspy.HighsLp:
    """Translate the IR to a ``HighsLp`` (column-wise sparse matrix)."""
    lp = highspy.HighsLp()
    lp.model_name_ = model.name
    lp.num_col_ = model.num_vars
    lp.num_row_ = model.num_constrs
    lp.col_cost_ = model.c
    lp.col_lower_ = model.lb
    lp.col_upper_ = model.ub
    lp.row_lower_, lp.row_upper_ = _row_bounds(model.senses, model.rhs)
    lp.offset_ = model.c0
    lp.sense_ = (
        highspy.ObjSense.kMinimize
        if model.sense is ObjSense.MINIMIZE
        else highspy.ObjSense.kMaximize
    )
    csc = model.A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = csc.data
    if model.is_mip:
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if is_int else highspy.HighsVarType.kContinuous
            for is_int in model.integer_mask
        ]
    lp.col_names_ = _lp_safe(model.var_names())
    lp.row_names_ = _lp_safe(model.constr_names())
    return lp


class HighsBackend(SolverBackend):
    """HiGHS: simplex / IPM for LPs, branch-and-cut for MIPs."""

    name: ClassVar[BackendName] = BackendName.HIGHS
    capabilities: ClassVar[frozenset[Capability]] = frozenset(
        {
            Capability.DUALS,
            Capability.WARM_START,
            Capability.EXPORT,
            Capability.PROGRESS,
            Capability.USER_CUTS,  # as a root cut loop before branch-and-bound
        }
    )
    #: Rounds of the root cut loop.
    MAX_ROOT_ROUNDS = 50

    def version(self) -> str:  # noqa: D102
        h = highspy.Highs()
        return f"{h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}"

    def solve(  # noqa: D102
        self,
        model: LinearModel,
        settings: SolverSettings,
        *,
        lazy: LazyOracle | None = None,
        warm_start: FloatArray | None = None,
        user_cuts: LazyOracle | None = None,
    ) -> RawSolution:
        h = highspy.Highs()
        h.setOptionValue("output_flag", settings.verbose)
        h.setOptionValue("mip_rel_gap", settings.mip_gap)
        h.setOptionValue("random_seed", settings.seed)
        if settings.threads:
            h.setOptionValue("threads", settings.threads)
        for name, value in settings.highs.items():
            if h.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise ValueError(f"invalid HiGHS option {name} = {value!r}")
        root_cuts = (
            self._root_cut_loop(model, settings, user_cuts) if user_cuts and model.is_mip else []
        )
        h.passModel(to_highs_lp(model.with_cuts(root_cuts, name="root_cuts")))
        if settings.export is not None:
            h.writeModel(str(settings.export))
        if warm_start is not None and not np.isnan(warm_start).any():
            start = highspy.HighsSolution()
            start.col_value = warm_start.tolist()
            start.value_valid = True
            h.setSolution(start)

        progress = ProgressRecorder()
        clock = {"offset": 0.0}  # HiGHS restarts its timer on every run
        minimise = model.sense is ObjSense.MINIMIZE
        best_bound = {"value": -np.inf if minimise else np.inf}
        if model.is_mip:

            def on_event(event: Any) -> None:
                out = event.data_out
                incumbent, bound = out.mip_primal_bound, out.mip_dual_bound
                if lazy is not None:
                    # Row generation: every round's bound is valid for the full
                    # model, so keep the best one; incumbents of a round may still
                    # violate lazy constraints and are not shown.
                    if np.isfinite(bound):
                        pick = max if minimise else min
                        best_bound["value"] = pick(best_bound["value"], bound)
                    incumbent, bound = float("inf"), best_bound["value"]
                progress.record(clock["offset"] + out.running_time, incumbent, bound)

            h.cbMipImprovingSolution.subscribe(on_event)
            h.cbMipInterrupt.subscribe(on_event)

        t0 = time.perf_counter()
        rounds = cuts_added = 0
        lazy_violated = False
        while True:
            clock["offset"] = time.perf_counter() - t0
            remaining = settings.time_limit - clock["offset"]
            h.setOptionValue("time_limit", max(remaining, 1e-3))
            h.run()
            if lazy is None or h.getModelStatus() != _MS.kOptimal:
                break
            x = np.asarray(h.getSolution().col_value, dtype=np.float64)
            cuts = list(lazy(x))
            lazy_violated = bool(cuts)
            if not cuts:
                break
            rounds += 1
            for cut in cuts:
                lo = -np.inf if cut.sense is Sense.LE else cut.rhs
                hi = np.inf if cut.sense is Sense.GE else cut.rhs
                h.addRow(lo, hi, len(cut.cols), cut.cols.astype(np.int32), cut.coefs)
                cuts_added += 1
            if time.perf_counter() - t0 >= settings.time_limit:
                break

        extra: dict[str, Any] = {"wall_time": time.perf_counter() - t0}
        if user_cuts is not None and model.is_mip:
            extra["user_cuts"] = len(root_cuts)
        if lazy is not None:
            extra.update(lazy_rounds=rounds, lazy_cuts=cuts_added)
        return self._collect(h, model, lazy_violated, extra, progress)

    def write_model(self, model: LinearModel, path: str) -> None:
        """Write the model with HiGHS's writer (``.lp``, ``.mps``)."""
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.passModel(to_highs_lp(model))
        h.writeModel(path)

    def _root_cut_loop(
        self, model: LinearModel, settings: SolverSettings, separator: LazyOracle
    ) -> list[Cut]:
        """Cutting planes from the LP relaxation: solve, separate, repeat.

        HiGHS has no user-cut callback, so the separator is applied to the root
        LP before branch-and-bound and the cuts become ordinary rows.
        """
        lp = highspy.Highs()
        lp.setOptionValue("output_flag", False)
        lp.passModel(to_highs_lp(model.relaxed()))
        cuts: list[Cut] = []
        for _ in range(self.MAX_ROOT_ROUNDS):
            lp.run()
            if lp.getModelStatus() != _MS.kOptimal:
                break
            new = list(separator(np.asarray(lp.getSolution().col_value, dtype=np.float64)))
            if not new:
                break
            for cut in new:
                lo = -np.inf if cut.sense is Sense.LE else cut.rhs
                hi = np.inf if cut.sense is Sense.GE else cut.rhs
                lp.addRow(lo, hi, len(cut.cols), cut.cols.astype(np.int32), cut.coefs)
            cuts.extend(new)
        return cuts

    def _collect(
        self,
        h: highspy.Highs,
        model: LinearModel,
        lazy_violated: bool,
        extra: dict[str, Any],
        progress: ProgressRecorder,
    ) -> RawSolution:
        status = _STATUS.get(h.getModelStatus(), SolveStatus.OTHER)
        info = h.getInfo()
        sol = h.getSolution()
        # A row-generation run stopped early may hold a point that violates
        # outstanding lazy constraints: it is not feasible for the true model.
        has_solution = info.primal_solution_status == _FEASIBLE and not lazy_violated
        if lazy_violated and status is SolveStatus.OPTIMAL:
            status = SolveStatus.TIME_LIMIT
        objective = float(info.objective_function_value) if has_solution else None
        bound: float | None = None
        duals = reduced = None
        if model.is_mip:
            if np.isfinite(info.mip_dual_bound):
                bound = float(info.mip_dual_bound)
        elif status is SolveStatus.OPTIMAL:
            bound = objective
            if sol.dual_valid:
                duals = np.asarray(sol.row_dual, dtype=np.float64)[: model.num_constrs]
                reduced = np.asarray(sol.col_dual, dtype=np.float64)
        runtime = float(extra["wall_time"]) if "lazy_rounds" in extra else float(h.getRunTime())
        if model.is_mip:  # close the trajectory with the final state
            inf = float("inf")
            progress.record(
                runtime, inf if objective is None else objective, inf if bound is None else bound
            )
        return RawSolution(
            status=status,
            backend=self.name.value,
            backend_version=self.version(),
            objective=objective,
            bound=bound,
            x=np.asarray(sol.col_value, dtype=np.float64) if has_solution else None,
            duals=duals,
            reduced_costs=reduced,
            runtime=runtime,
            iterations=int(max(info.simplex_iteration_count, 0)),
            nodes=int(max(info.mip_node_count, 0)) if model.is_mip else 0,
            progress=tuple(progress.points),
            extra=extra,
        )
