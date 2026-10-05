# Dashboard

```bash
uv sync --extra app --extra highs   # plus --extra gurobi
uv run linopt app
```

The dashboard is a thin Streamlit layer over the same objects the CLI uses:
presets are the TOML files in `configs/`, widgets are generated from each
model's pydantic options, and the *Model & export* tab writes the current
settings back as a TOML config, so every interactive run is reproducible from
the command line.

::: linear_opt.app.logic
