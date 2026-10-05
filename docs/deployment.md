# Deployment

## Docker

```bash
docker build -t linear-opt .
docker run --rm linear-opt solve configs/jobshop_ft06.toml
docker run --rm -p 8501:8501 linear-opt app --headless --address 0.0.0.0
docker compose up          # dashboard with health check
```

The image installs the project as a wheel (benchmark snapshots included),
runs as uid 1000, and reads presets from `/app/configs`
(`LINEAR_OPT_CONFIG_DIR`). Mount a licence file and set `GRB_LICENSE_FILE` for
a full Gurobi licence; otherwise the restricted pip licence is used and HiGHS
is always available.

## Continuous integration

* `ci.yml` - lint, types, lockfile, tests on 3.11-3.13 + macOS with a 90 %
  coverage gate, docs build, Docker smoke tests (benchmarks solved in the
  image, dashboard health check).
* `large.yml` - instances above the restricted licence, with a WLS licence
  stored as repository secrets.
* `release.yml` - on `vX.Y.Z` tags: version check, CI, wheel/sdist, multi-arch
  GHCR image, GitHub release with notes from `CHANGELOG.md`.
* `docs.yml` - GitHub Pages.
