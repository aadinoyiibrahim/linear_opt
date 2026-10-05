# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). The release workflow uses the
section matching the tag as the GitHub release notes.

## [Unreleased]

## [0.1.0] - 2026-10-02

### Added

- Solver-neutral modelling layer (`ModelBuilder` -> `LinearModel`) with Gurobi
  (matrix API, lazy constraints, user cuts, MIP starts, IIS, progress callbacks)
  and HiGHS backends (root cut loop, row generation, progress callbacks).
- Problem classes, each validated against published optima:
  - transportation / discrete optimal transport (LP, with duality checks);
  - capacitated facility location (strong vs weak formulations; OR-Library cap41);
  - TSP (DFJ with exact separation, single-commodity flow, MTZ; 14 TSPLIB instances);
  - CVRP (two-index with rounded capacity cuts, MTZ load; CVRPLIB A-n32-k5);
  - job-shop scheduling (disjunctive with tight/naive big-M, time-indexed; JSPLIB
    ft06, la01-la05).
- Formulation studies (`linopt variants`): LP bound, root gap, runtime, nodes.
- Interactive Plotly reports and a Streamlit dashboard (`linopt app`).
- TOML run configurations validated with pydantic; reproducible exports from
  the dashboard.
- Data loaders with provenance (`sources.toml`): GeoNames, OSRM, OR-Library,
  TSPLIB, CVRPLIB, JSPLIB; packaged benchmark snapshots.
- Docker image (CLI + dashboard), docker compose, CI (lint, types, tests on
  Python 3.11-3.13 and macOS, docs, Docker smoke tests), release workflow
  (GHCR multi-arch image, wheel/sdist), docs on GitHub Pages.

[Unreleased]: https://github.com/aadinoyiibrahim/linear_opt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aadinoyiibrahim/linear_opt/releases/tag/v0.1.0
