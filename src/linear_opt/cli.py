
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from linear_opt import __version__
from linear_opt.core.backends import available_backends, resolve_backend
from linear_opt.core.config import BackendName, ConfigError, RunConfig, load_config
from linear_opt.core.model import OptimizationModel, Result
from linear_opt.core.result import SolveStatus

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Download and snapshot datasets.")
app.add_typer(data_app, name="data")
console = Console()
err = Console(stderr=True)

ConfigArg = Annotated[Path, typer.Argument(help="TOML run config.", exists=True, dir_okay=False)]
BackendOpt = Annotated[
    BackendName | None,
    typer.Option("--backend", "-b", help="Override [solver].backend."),
]


def _load(path: Path, **solver_overrides: Any) -> RunConfig:
    try:
        cfg = load_config(path)
    except ConfigError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    overrides = {k: v for k, v in solver_overrides.items() if v is not None}
    if overrides:
        cfg = cfg.model_copy(update={"solver": cfg.solver.model_copy(update=overrides)})
    return cfg


def _build_model(cfg: RunConfig) -> OptimizationModel[Any, Any, Any]:
    from linear_opt.problems.registry import model_from_config

    try:
        return model_from_config(cfg)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _summary_table(result: Result[Any]) -> Table:
    s = result.summary()
    table = Table(title=f"{s['name']} ({s['kind']})", show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    colour = "green" if result.ok else "yellow" if result.raw.has_solution else "red"
    table.add_row("status", f"[{colour}]{s['status']}[/{colour}]")
    for key in ("objective", "bound", "gap", "runtime_s", "iterations", "nodes", "backend"):
        value = s.get(key)
        if value is not None:
            table.add_row(key, f"{value:,.6g}" if isinstance(value, float) else str(value))
    if s.get("reference") is not None:
        table.add_row("known optimum", f"{s['reference']:,.6g}")
        dev = s["reference_gap"]
        ok = dev is not None and abs(dev) <= 1e-6
        table.add_row("deviation", f"[{'green' if ok else 'yellow'}]{dev:.2e}[/]")
    m = s["model"]
    table.add_row(
        "size", f"{m['variables']} vars · {m['constraints']} constrs · {m['nonzeros']} nnz"
    )
    if result.violations:
        table.add_row("violations", "\n".join(result.violations[:5]))
    return table


def _figures(model: OptimizationModel[Any, Any, Any], result: Result[Any], dark: bool) -> list[Any]:
    from linear_opt.viz import figures_for

    return figures_for(model.kind.value, model.data, result.solution, dark=dark)


def _report_module() -> Any:
    try:
        from linear_opt.viz import report
    except ImportError as exc:
        err.print("[red]HTML reports need the viz extra: uv sync --extra viz[/red]")
        raise typer.Exit(2) from exc
    return report


@app.command()
def info() -> None:
    """Show the package version and the installed solver backends."""
    console.print(f"linear-opt {__version__}")
    backends = available_backends()
    if not backends:
        console.print("backends: none installed (uv sync --extra highs)")
    for name, ver in backends.items():
        console.print(f"  {name:<7} {ver}")


@app.command()
def validate(
    configs: Annotated[list[Path], typer.Argument(help="One or more TOML run configs.")],
) -> None:
    """Validate run configurations without solving."""
    failed = False
    for path in configs:
        try:
            cfg = load_config(path)
        except ConfigError as exc:
            failed = True
            err.print(f"[red]FAIL[/red] {path}\n{exc}")
            continue
        console.print(f"[green]ok[/green]   {path}  ({cfg.problem.kind}: {cfg.problem.name})")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def backend(config: ConfigArg) -> None:
    """Print which backend a config would use on this machine."""
    console.print(resolve_backend(_load(config).solver.backend).value)


@app.command()
def solve(
    config: ConfigArg,
    backend: BackendOpt = None,
    time_limit: Annotated[float | None, typer.Option(help="Seconds.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show solver log.")] = False,
    html: Annotated[Path | None, typer.Option(help="Write an interactive HTML report.")] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the summary as JSON.")
    ] = None,
    export: Annotated[Path | None, typer.Option(help="Write the model (.lp/.mps).")] = None,
    dark: Annotated[bool, typer.Option(help="Dark theme for the HTML report.")] = False,
    relax: Annotated[
        bool, typer.Option("--relax", help="Solve only the LP relaxation (drop integrality).")
    ] = False,
) -> None:
    """Build and solve the model described by CONFIG."""
    from linear_opt.solvers import Capability, LicenseLimitError, create_backend

    cfg = _load(
        config, backend=backend, time_limit=time_limit, verbose=verbose or None, export=export
    )
    model = _build_model(cfg)
    solver = create_backend(cfg.solver.backend)
    if relax:
        raw = solver.solve(model.build().relaxed(), cfg.solver)
        console.print(f"LP relaxation of {cfg.problem.name}: {raw.status.value}, ", end="")
        console.print("no solution" if raw.objective is None else f"objective {raw.objective:,.6f}")
        reference = model.reference_objective()
        if raw.objective is not None and reference is not None:
            gap = abs(reference - raw.objective) / max(abs(reference), 1e-10)
            console.print(f"root gap to known optimum {reference:,.6f}: {100 * gap:.3f}%")
        return
    try:
        result = model.solve(cfg.solver, backend=solver)
    except LicenseLimitError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc
    console.print(_summary_table(result))

    if result.raw.status in (SolveStatus.INFEASIBLE, SolveStatus.INF_OR_UNBD):
        if solver.supports(Capability.IIS):
            iis = solver.compute_iis(model.build(), cfg.solver)
            console.print("[bold]Irreducible infeasible subsystem:[/bold]")
            for name in iis[:25]:
                console.print(f"  {name}")
            if len(iis) > 25:
                console.print(f"  ... and {len(iis) - 25} more")
        else:
            console.print("[dim]Tip: `--backend gurobi` explains infeasibility with an IIS.[/dim]")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
        console.print(f"summary  -> {json_out}")
    if html is not None:
        report = _report_module()
        summary = result.summary()
        report.write_report(
            html,
            f"{cfg.problem.name} · {cfg.problem.kind}",
            _figures(model, result, dark),
            kpis=report.run_kpis(summary),
            subtitle=report.model_subtitle(summary),
        )
        console.print(f"report   -> {html}")
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def compare(config: ConfigArg) -> None:
    """Solve CONFIG with every installed backend and compare the results."""
    cfg = _load(config)
    table = Table(title=f"{cfg.problem.name}: backend comparison")
    for col in ("backend", "status", "objective", "bound", "runtime (s)", "valid"):
        left = col in ("backend", "status")
        table.add_column(col, justify="left" if left else "right", overflow="fold")
    objectives: list[float] = []
    for name in available_backends():
        result = _build_model(cfg).solve(cfg.solver, backend=name)
        s = result.summary()
        if result.raw.objective is not None:
            objectives.append(result.raw.objective)
        table.add_row(
            s["backend"],
            s["status"],
            "—" if s["objective"] is None else f"{s['objective']:,.6f}",
            "—" if s["bound"] is None else f"{s['bound']:,.6f}",
            f"{s['runtime_s']:.3f}",
            "yes" if result.ok else "no",
        )
    console.print(table)
    if len(objectives) > 1:
        spread = (max(objectives) - min(objectives)) / max(1.0, abs(objectives[0]))
        verdict = (
            "[green]agree[/green]" if spread < 1e-6 else f"[yellow]differ by {spread:.2e}[/yellow]"
        )
        console.print(f"objectives {verdict}")


@app.command()
def variants(
    config: ConfigArg,
    backend: BackendOpt = None,
    html: Annotated[Path | None, typer.Option(help="Write an HTML comparison report.")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write rows as JSON.")] = None,
) -> None:
    """Compare the model's formulations: LP bound, root gap, runtime and nodes."""
    from linear_opt.core.study import run_variants

    cfg = _load(config, backend=backend)
    model = _build_model(cfg)
    if not type(model).variants():
        err.print(f"[yellow]{model.kind} defines no variants[/yellow]")
        raise typer.Exit(2)
    from linear_opt.solvers import LicenseLimitError

    try:
        results = run_variants(model, cfg.solver)
    except LicenseLimitError as exc:
        err.print(f"[red]{exc}[/red]\n[dim]Compact formulations (MTZ, flow) grow as n^2.[/dim]")
        raise typer.Exit(3) from exc
    reference = model.reference_objective()
    title = f"{cfg.problem.name}: formulations ({resolve_backend(cfg.solver.backend).value})"
    table = Table(title=title)
    headers = (
        "variant",
        "LP bound",
        "root gap",
        "objective",
        "runtime (s)",
        "nodes",
        "rows",
        "valid",
    )
    for col in headers:
        if col == "variant":
            table.add_column(col, justify="left", no_wrap=True)  # names stay readable
        else:
            table.add_column(col, justify="right", overflow="fold")
    for r in results:
        row = r.row()
        table.add_row(
            row["variant"],
            "—" if row["lp_bound"] is None else f"{row['lp_bound']:,.2f}",
            "—" if row["root_gap"] is None else f"{100 * row['root_gap']:.3f}%",
            "—" if row["objective"] is None else f"{row['objective']:,.3f}",
            f"{row['runtime_s']:.3f}",
            f"{row['nodes']:,}",
            f"{row['constraints']:,}",
            "yes" if row["valid"] else "no",
        )
    console.print(table)
    if reference is not None:
        console.print(f"known optimum: {reference:,.3f}")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps([r.row() for r in results], indent=2), encoding="utf-8")
        console.print(f"rows     -> {json_out}")
    if html is not None:
        report = _report_module()
        from linear_opt.viz import mip

        figs = [mip.variant_root_gaps(results), *mip.variant_effort(results)]
        figs.append(mip.variant_progress(results))
        kpis: dict[str, Any] = {
            f"{r.name}: root gap": None if r.root_gap is None else f"{100 * r.root_gap:.2f}%"
            for r in results
        }
        if reference is not None:
            kpis["Known optimum"] = reference
        report.write_report(html, title, figs, kpis=kpis, subtitle=str(config))
        console.print(f"report   -> {html}")


@app.command("app")
def run_app(
    port: Annotated[int, typer.Option(help="Port of the web server.")] = 8501,
    headless: Annotated[bool, typer.Option(help="Do not open a browser window.")] = False,
    address: Annotated[
        str | None, typer.Option(help="Interface to bind, e.g. 0.0.0.0 inside Docker.")
    ] = None,
) -> None:
    """Start the interactive dashboard (needs the `app` extra)."""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        err.print("[red]The dashboard needs the app extra: uv sync --extra app[/red]")
        raise typer.Exit(2)
    main = Path(__file__).with_name("app") / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(main), "--server.port", str(port)]
    if headless:
        cmd += ["--server.headless", "true"]
    if address:
        cmd += ["--server.address", address]
    raise typer.Exit(subprocess.call(cmd))


@data_app.command("orlib-cap")
def data_orlib_cap(
    ids: Annotated[
        list[str] | None, typer.Argument(help="Instances, e.g. cap41 cap131 (default: cap41).")
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="All 37 instances.")] = False,
    out: Annotated[
        Path | None, typer.Option(help="Directory (default: packaged fixtures).")
    ] = None,
) -> None:
    """Download OR-Library capacitated warehouse instances plus capopt.txt."""
    from linear_opt.data import orlib
    from linear_opt.data.download import DownloadError

    names = list(orlib.CAP_IDS) if all_ else (ids or ["cap41"])
    unknown = sorted(set(names) - set(orlib.CAP_IDS))
    if unknown:
        err.print(f"[red]unknown instances: {', '.join(unknown)}[/red]")
        raise typer.Exit(2)
    try:
        written = orlib.snapshot(names, out)
    except (DownloadError, ValueError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"wrote {len(written)} files -> {written[0].parent}")
    optima = orlib.parse_capopt((written[0].parent / orlib.OPT_FILE).read_text())
    for name in names:
        console.print(f"  {name:<7} known optimum {optima.get(name, float('nan')):,.3f}")


@data_app.command("tsplib")
def data_tsplib(
    names: Annotated[
        list[str] | None, typer.Argument(help="Instances, e.g. berlin52 eil51 (default: berlin52).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Directory (default: packaged fixtures).")
    ] = None,
    from_dir: Annotated[
        Path | None, typer.Option(help="Import <name>.tsp/.opt.tour you downloaded manually.")
    ] = None,
) -> None:
    """Download TSPLIB instances (.tsp) and their optimal tours (.opt.tour) if published."""
    from linear_opt.data import tsplib
    from linear_opt.data.download import DownloadError

    for name in names or ["berlin52"]:
        try:
            written = tsplib.snapshot_tsplib(name, out, from_dir=from_dir)
        except (DownloadError, ValueError) as exc:
            err.print(f"[red]{name}: {exc}[/red]")
            raise typer.Exit(1) from exc
        problem, optimum, tour = tsplib.load_tsplib(name) if out is None else (None, None, None)
        files = ", ".join(p.name for p in written)
        console.print(f"{name:<10} {files}")
        if problem is not None:
            length = tsplib.tour_length(problem.distances, tour) if tour else None
            console.print(f"           published optimum {optimum}, optimal tour length {length}")


@data_app.command("cvrplib")
def data_cvrplib(
    names: Annotated[
        list[str] | None, typer.Argument(help="Instances, e.g. A-n32-k5 (default: A-n32-k5).")
    ] = None,
    from_dir: Annotated[
        Path | None, typer.Option(help="Import <name>.vrp/.sol you downloaded manually.")
    ] = None,
    base_url: Annotated[str | None, typer.Option(help="CVRPLIB site root.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Directory (default: packaged fixtures).")
    ] = None,
) -> None:
    """Fetch CVRPLIB instances (.vrp) and optimal solutions (.sol)."""
    from linear_opt.data import tsplib
    from linear_opt.data.download import DownloadError

    for name in names or ["A-n32-k5"]:
        try:
            written = tsplib.snapshot_cvrplib(
                name, out, base_url=base_url or tsplib.CVRPLIB_URL, from_dir=from_dir
            )
        except (DownloadError, FileNotFoundError, ValueError) as exc:
            err.print(f"[red]{name}: {exc}[/red]")
            err.print(
                "[dim]Download the .vrp and .sol files from CVRPLIB "
                "(https://galgos.inf.puc-rio.br/cvrplib/en/instances) "
                "and re-run with --from-dir.[/dim]"
            )
            raise typer.Exit(1) from exc
        console.print(f"{name:<10} {', '.join(p.name for p in written)}")


@data_app.command("jsplib")
def data_jsplib(
    names: Annotated[
        list[str] | None,
        typer.Argument(help="Instances, e.g. ft06 la01 (default: ft06 la01-la05)."),
    ] = None,
    orlib_file: Annotated[
        Path | None, typer.Option(help="Import from a local OR-Library jobshop1.txt instead.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Directory (default: packaged fixtures).")
    ] = None,
) -> None:
    """Download job-shop instances and their optima/bounds from JSPLIB."""
    from linear_opt.data import jsplib
    from linear_opt.data.download import DownloadError

    chosen = names or ["ft06", "la01", "la02", "la03", "la04", "la05"]
    try:
        written = jsplib.snapshot(chosen, out, orlib_file=orlib_file)
    except (DownloadError, ValueError, OSError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"wrote {len(written)} files -> {written[0].parent}")
    if out is None and orlib_file is None:
        for name in chosen:
            ref = jsplib.reference(name)
            if ref is None:
                continue
            value = (
                f"optimum {ref.optimum:g}"
                if ref.optimum
                else f"bounds [{ref.lower:g}, {ref.upper:g}]"
            )
            console.print(f"  {name:<7} {value}")


@data_app.command("cities")
def data_cities(
    out: Annotated[
        Path | None,
        typer.Option(help="CSV to write (default: the packaged fixture)."),
    ] = None,
    country: Annotated[str, typer.Option(help="ISO country code.")] = "DE",
    limit: Annotated[int, typer.Option(help="Keep the N largest cities.")] = 100,
) -> None:
    """Download GeoNames cities15000 and freeze a snapshot as CSV."""
    from linear_opt.data import geonames
    from linear_opt.data.download import DownloadError

    try:
        cities = geonames.download_cities(country)[:limit]
    except DownloadError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    target = out or geonames.fixture_path()
    geonames.write_cities(cities, target)
    console.print(f"wrote {len(cities)} cities -> {target}")


if __name__ == "__main__":  # pragma: no cover
    app()
