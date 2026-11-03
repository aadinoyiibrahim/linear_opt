# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). The release workflow uses the
section matching the tag as the GitHub release notes.

## [Unreleased]

## [1.0.0] - 2026-11-03

First public release.

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
- Formulation studies (`linearmilpopt variants`): LP bound, root gap, runtime, nodes.
- Interactive Plotly reports and a Streamlit dashboard (`linearmilpopt app`).
- TOML run configurations validated with pydantic; reproducible exports from
  the dashboard.
- Data loaders with provenance (`sources.toml`): GeoNames, OSRM, OR-Library,
  TSPLIB, CVRPLIB, JSPLIB; packaged benchmark snapshots.
- Docker image (CLI + dashboard), docker compose, CI (lint, types, tests on
  Python 3.11-3.13 and macOS, docs, Docker smoke tests), release workflow
  (GHCR multi-arch image, wheel/sdist), docs on GitHub Pages.

### Changed

- **Renamed the project to LinearMilpOpt** (pre-release name: linear_opt). If you
  installed a pre-release:
  - package and import name `linear_opt` / `linear-opt` -> `linearmilpopt`;
  - command `linopt` -> `linearmilpopt`;
  - environment variables `LINEAR_OPT_DATA_DIR` / `LINEAR_OPT_CONFIG_DIR` ->
    `LINEARMILPOPT_DATA_DIR` / `LINEARMILPOPT_CONFIG_DIR`;
  - download cache `~/.cache/linear_opt` -> `~/.cache/linearmilpopt`;
  - repository `aadinoyiibrahim/LinearMilpOpt`, image
    `ghcr.io/aadinoyiibrahim/linearmilpopt`, dashboard
    `linearmilpopt.streamlit.app`.

[Unreleased]: https://github.com/aadinoyiibrahim/LinearMilpOpt/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/aadinoyiibrahim/LinearMilpOpt/releases/tag/v1.0.0
