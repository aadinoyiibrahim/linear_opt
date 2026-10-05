# linear_opt

Production-grade LP/MILP modelling in Python: **Gurobi** as the primary solver,
**HiGHS** as an open-source fallback, and real benchmark instances with known optima.

| Problem | Class | Benchmark | Known optimum |
|---|---|---|---|
| Transport / optimal transport | LP | GeoNames + OSM (Germany) | — |
| Capacitated facility location | MILP | OR-Library cap41 | 1 040 444.375 (verified from capopt.txt) |
| Travelling salesman | MILP + lazy cuts | TSPLIB berlin52 | 7 542 (TSPLIB STSP table) |
| Capacitated vehicle routing | MILP + lazy cuts | CVRPLIB A-n32-k5 | 784 (CVRPLIB .sol) |
| Job-shop scheduling | MILP | JSPLIB ft06, la01–la05 | 55; 666, 655, 597, 590, 593 (JSPLIB) |

```bash
uv sync --extra highs          # or --extra gurobi, or --all-extras
uv run linopt info
uv run linopt validate configs/*.toml
```
