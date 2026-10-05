r"""LP duality checks: an independent certificate of optimality.

For a primal-dual pair :math:`(x^*, y^*)` with reduced costs
:math:`z = c - A^\top y`, the Lagrangian dual objective is

.. math::

    g(y) = c_0 + b^\top y + \sum_j z_j \cdot
        \begin{cases} \ell_j & z_j > 0 \\ u_j & z_j < 0 \end{cases}
    \qquad (\text{minimisation; bounds swap for maximisation}).

Strong duality says :math:`g(y^*) = c^\top x^* + c_0`. Checking it, plus the
sign conditions on :math:`y`, verifies a solver's answer without trusting it.
"""

from __future__ import annotations

import numpy as np

from linear_opt.core.ir import FloatArray, LinearModel, ObjSense, Sense


def dual_objective(model: LinearModel, y: FloatArray, tol: float = 1e-9) -> float:
    """Lagrangian dual objective ``g(y)``; ``-inf``/``+inf`` if ``y`` is dual infeasible."""
    z = model.c - model.A.T @ y
    minimise = model.sense is ObjSense.MINIMIZE
    at_lower = z > tol if minimise else z < -tol
    at_upper = z < -tol if minimise else z > tol
    bound = np.zeros_like(z)
    bound[at_lower] = model.lb[at_lower]
    bound[at_upper] = model.ub[at_upper]
    if not np.all(np.isfinite(bound)):
        return -np.inf if minimise else np.inf
    return float(model.c0 + model.rhs @ y + z @ bound)


def dual_sign_violations(model: LinearModel, y: FloatArray, tol: float = 1e-7) -> list[str]:
    """Rows whose dual has the wrong sign for the row sense."""
    flip = 1.0 if model.sense is ObjSense.MINIMIZE else -1.0
    problems: list[str] = []
    names = model.constr_names()
    for i, (sense, value) in enumerate(zip(model.senses, y * flip, strict=True)):
        if (sense == Sense.GE.value and value < -tol) or (sense == Sense.LE.value and value > tol):
            problems.append(f"{names[i]}: dual {y[i]:.3g} has wrong sign for '{sense}'")
    return problems


def duality_gap(model: LinearModel, primal_objective: float, y: FloatArray) -> float:
    """Relative gap between primal objective and dual objective."""
    return abs(primal_objective - dual_objective(model, y)) / max(1.0, abs(primal_objective))
