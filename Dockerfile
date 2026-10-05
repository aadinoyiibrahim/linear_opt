# syntax=docker/dockerfile:1.7
# One image for the CLI and the dashboard:
#   docker run --rm IMAGE solve configs/jobshop_ft06.toml
#   docker run --rm -p 8501:8501 IMAGE app --headless --address 0.0.0.0
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- builder
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first (cached layer), project code second.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --extra gurobi --extra highs --extra app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra gurobi --extra highs --extra app

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
LABEL org.opencontainers.image.source="https://github.com/aadinoyiibrahim/linear_opt" \
      org.opencontainers.image.description="LP/MILP modelling with Gurobi and HiGHS: CLI and dashboard" \
      org.opencontainers.image.licenses="MIT"

RUN useradd --create-home --uid 1000 opt && mkdir -p /app && chown opt:opt /app
WORKDIR /app
COPY --from=builder --chown=opt:opt /app/.venv /app/.venv
COPY --chown=opt:opt configs ./configs
COPY --chown=opt:opt .streamlit ./.streamlit

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LINEAR_OPT_CONFIG_DIR=/app/configs \
    LINEAR_OPT_DATA_DIR=/home/opt/.cache/linear_opt \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Full Gurobi licence (optional): mount a licence file and point to it, e.g.
#   -v $HOME/gurobi.lic:/opt/gurobi/gurobi.lic:ro -e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
# Without it gurobipy uses its size-restricted licence; HiGHS is always available.
USER 1000:1000
EXPOSE 8501
ENTRYPOINT ["linopt"]
CMD ["--help"]
