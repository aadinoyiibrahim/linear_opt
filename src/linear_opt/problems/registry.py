"""Turn a :class:`~linear_opt.core.config.RunConfig` into a ready-to-solve model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from linear_opt.core.config import InstanceSettings, ProblemKind, RunConfig
from linear_opt.core.model import OptimizationModel
from linear_opt.data import geonames, jsplib, orlib, tsplib
from linear_opt.problems.cvrp import CVRPInstance, CVRPModel
from linear_opt.problems.facility import FacilityInstance, FacilityModel
from linear_opt.problems.jobshop import JobShopInstance, JobShopModel
from linear_opt.problems.transport import TransportInstance, TransportModel
from linear_opt.problems.tsp import TSPInstance, TSPModel


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticTransportParams(_Params):
    """``[instance.params]`` for ``source = "synthetic"``."""

    n_sources: int = Field(default=10, gt=0)
    n_sinks: int = Field(default=25, gt=0)
    slack: float = Field(default=0.1, ge=0)


class CityTransportParams(_Params):
    """``[instance.params]`` for ``source = "geonames-de"``."""

    n_cities: int = Field(default=40, ge=3)
    n_sources: int = Field(default=8, gt=0)
    slack: float = Field(default=0.1, ge=0)
    cost: Literal["haversine", "road"] = "haversine"


def _cities(inst: InstanceSettings, n: int) -> list[geonames.City]:
    if inst.id == "fixture":
        cities = geonames.load_fixture()
    elif inst.id == "live":
        cities = geonames.download_cities("DE")
    else:
        raise ValueError(f"geonames-de: unknown id {inst.id!r} (use 'fixture' or 'live')")
    if len(cities) < n:
        raise ValueError(f"only {len(cities)} cities available, {n} requested")
    return cities[:n]


def load_transport_instance(inst: InstanceSettings) -> TransportInstance:
    """Load or generate a transport instance described by ``[instance]``."""
    if inst.source == "synthetic":
        sp = SyntheticTransportParams.model_validate(inst.params)
        return TransportInstance.random(sp.n_sources, sp.n_sinks, seed=inst.seed, slack=sp.slack)
    if inst.source == "geonames-de":
        cp = CityTransportParams.model_validate(inst.params)
        return TransportInstance.from_cities(
            _cities(inst, cp.n_cities),
            n_sources=cp.n_sources,
            seed=inst.seed,
            slack=cp.slack,
            cost=cp.cost,
        )
    raise ValueError(f"transport: unsupported instance source {inst.source!r}")


class SyntheticFacilityParams(_Params):
    """``[instance.params]`` for synthetic facility location."""

    n_facilities: int = Field(default=15, gt=0)
    n_customers: int = Field(default=50, gt=0)
    capacity_ratio: float = Field(default=1.6, gt=1.0)


class CityFacilityParams(_Params):
    """``[instance.params]`` for facility location on German cities."""

    n_cities: int = Field(default=60, ge=2)
    n_candidates: int = Field(default=20, gt=0)
    capacity_ratio: float = Field(default=2.5, gt=1.0)
    fixed_cost: float = Field(default=50_000.0, ge=0)
    unit_cost: float = Field(default=1.0, ge=0)


def load_facility_instance(inst: InstanceSettings) -> FacilityInstance:
    """Load or generate a facility-location instance described by ``[instance]``."""
    if inst.source == "orlib-cap":
        assert inst.id is not None
        return FacilityInstance.from_orlib(orlib.load_cap(inst.id), orlib.known_optimum(inst.id))
    if inst.source == "synthetic":
        sp = SyntheticFacilityParams.model_validate(inst.params)
        return FacilityInstance.random(
            sp.n_facilities, sp.n_customers, seed=inst.seed, capacity_ratio=sp.capacity_ratio
        )
    if inst.source == "geonames-de":
        cp = CityFacilityParams.model_validate(inst.params)
        return FacilityInstance.from_cities(
            _cities(inst, cp.n_cities),
            n_candidates=cp.n_candidates,
            capacity_ratio=cp.capacity_ratio,
            fixed_cost=cp.fixed_cost,
            unit_cost=cp.unit_cost,
        )
    raise ValueError(f"facility_location: unsupported instance source {inst.source!r}")


class SyntheticTSPParams(_Params):
    """``[instance.params]`` for a random TSP."""

    n: int = Field(default=30, ge=3)


class CityTSPParams(_Params):
    """``[instance.params]`` for a round trip through German cities."""

    n_cities: int = Field(default=40, ge=3)


def load_tsp_instance(inst: InstanceSettings) -> TSPInstance:
    """Load or generate a TSP instance described by ``[instance]``."""
    if inst.source == "tsplib":
        assert inst.id is not None
        problem, optimum, tour = tsplib.load_tsplib(inst.id)
        return TSPInstance.from_tsplib(problem, optimum, tour)
    if inst.source == "synthetic":
        return TSPInstance.random(SyntheticTSPParams.model_validate(inst.params).n, seed=inst.seed)
    if inst.source == "geonames-de":
        cp = CityTSPParams.model_validate(inst.params)
        return TSPInstance.from_cities(_cities(inst, cp.n_cities))
    raise ValueError(f"tsp: unsupported instance source {inst.source!r}")


class SyntheticCVRPParams(_Params):
    """``[instance.params]`` for a random CVRP."""

    n_customers: int = Field(default=15, ge=1)
    capacity: float = Field(default=100.0, gt=0)
    spare_vehicles: int = Field(default=1, ge=0)


class CityCVRPParams(_Params):
    """``[instance.params]`` for deliveries between German cities."""

    n_cities: int = Field(default=25, ge=2)
    depot: str = "Kassel"
    capacity: float = Field(default=150.0, gt=0)
    vehicles: int = Field(default=6, ge=1)


def load_cvrp_instance(inst: InstanceSettings) -> CVRPInstance:
    """Load or generate a CVRP instance described by ``[instance]``."""
    if inst.source == "cvrplib":
        assert inst.id is not None
        problem, optimum, routes = tsplib.load_cvrplib(inst.id)
        return CVRPInstance.from_tsplib(problem, optimum, routes)
    if inst.source == "synthetic":
        sp = SyntheticCVRPParams.model_validate(inst.params)
        return CVRPInstance.random(
            sp.n_customers, seed=inst.seed, capacity=sp.capacity, spare_vehicles=sp.spare_vehicles
        )
    if inst.source == "geonames-de":
        cp = CityCVRPParams.model_validate(inst.params)
        pool = _cities(inst, len(geonames.load_fixture()) if inst.id == "fixture" else 1000)
        chosen = pool[: cp.n_cities]
        if cp.depot not in {c.name for c in chosen}:
            depot = [c for c in pool if c.name == cp.depot]
            if not depot:
                raise ValueError(f"depot {cp.depot!r} not found among the cities")
            chosen = [*chosen[: cp.n_cities - 1], *depot]
        return CVRPInstance.from_cities(
            chosen, depot=cp.depot, capacity=cp.capacity, vehicles=cp.vehicles
        )
    raise ValueError(f"cvrp: unsupported instance source {inst.source!r}")


class SyntheticJobShopParams(_Params):
    """``[instance.params]`` for a random job shop."""

    n_jobs: int = Field(default=6, ge=1)
    n_machines: int = Field(default=4, ge=1)
    max_duration: int = Field(default=10, ge=1)


def load_jobshop_instance(inst: InstanceSettings) -> JobShopInstance:
    """Load or generate a job-shop instance described by ``[instance]``."""
    if inst.source == "jsplib":
        assert inst.id is not None
        data, ref = jsplib.load_jsp(inst.id)
        return JobShopInstance.from_data(data, ref)
    if inst.source == "synthetic":
        sp = SyntheticJobShopParams.model_validate(inst.params)
        return JobShopInstance.random(
            sp.n_jobs, sp.n_machines, seed=inst.seed, max_duration=sp.max_duration
        )
    if inst.path is not None:
        data = jsplib.parse_jsp(inst.path.read_text(encoding="utf-8"), inst.path.stem)
        return JobShopInstance.from_data(data)
    raise ValueError(f"job_shop: unsupported instance source {inst.source!r}")


def model_from_config(cfg: RunConfig) -> OptimizationModel[Any, Any, Any]:
    """Instantiate the model a config describes.

    Raises:
        NotImplementedError: For problem classes scheduled for later phases.
    """
    if cfg.problem.kind is ProblemKind.TRANSPORT:
        return TransportModel(
            load_transport_instance(cfg.instance), cfg.model, name=cfg.problem.name
        )
    if cfg.problem.kind is ProblemKind.TSP:
        return TSPModel(load_tsp_instance(cfg.instance), cfg.model, name=cfg.problem.name)
    if cfg.problem.kind is ProblemKind.CVRP:
        return CVRPModel(load_cvrp_instance(cfg.instance), cfg.model, name=cfg.problem.name)
    if cfg.problem.kind is ProblemKind.FACILITY_LOCATION:
        return FacilityModel(load_facility_instance(cfg.instance), cfg.model, name=cfg.problem.name)
    if cfg.problem.kind is ProblemKind.JOB_SHOP:
        return JobShopModel(load_jobshop_instance(cfg.instance), cfg.model, name=cfg.problem.name)
    raise NotImplementedError(f"problem kind '{cfg.problem.kind}' is not implemented yet")


# ------------------------------------------------------------------ catalogue
#: Model class per problem kind.
MODEL_CLASSES: dict[ProblemKind, type[OptimizationModel[Any, Any, Any]]] = {
    ProblemKind.TRANSPORT: TransportModel,
    ProblemKind.FACILITY_LOCATION: FacilityModel,
    ProblemKind.TSP: TSPModel,
    ProblemKind.CVRP: CVRPModel,
    ProblemKind.JOB_SHOP: JobShopModel,
}

#: Instance sources per problem kind, and the pydantic model of their
#: ``[instance.params]`` (``None``: the source takes no parameters).
SOURCES: dict[ProblemKind, dict[str, type[BaseModel] | None]] = {
    ProblemKind.TRANSPORT: {
        "synthetic": SyntheticTransportParams,
        "geonames-de": CityTransportParams,
    },
    ProblemKind.FACILITY_LOCATION: {
        "orlib-cap": None,
        "synthetic": SyntheticFacilityParams,
        "geonames-de": CityFacilityParams,
    },
    ProblemKind.TSP: {
        "tsplib": None,
        "synthetic": SyntheticTSPParams,
        "geonames-de": CityTSPParams,
    },
    ProblemKind.CVRP: {
        "cvrplib": None,
        "synthetic": SyntheticCVRPParams,
        "geonames-de": CityCVRPParams,
    },
    ProblemKind.JOB_SHOP: {
        "jsplib": None,
        "synthetic": SyntheticJobShopParams,
    },
}


def available_ids(source: str) -> list[str]:
    """Instance ids that can be loaded without a download (packaged snapshots)."""
    if source == "synthetic":
        return ["uniform"]
    if source == "geonames-de":
        return ["fixture"] if geonames.fixture_path().exists() else []
    if source == "orlib-cap":
        found = {p.stem for p in orlib.fixture_dir().glob("cap*.txt")}
        return [c for c in orlib.CAP_IDS if c in found]
    if source == "tsplib":
        return sorted(p.stem for p in tsplib.fixture_dir("tsplib").glob("*.tsp"))
    if source == "cvrplib":
        return sorted(p.stem for p in tsplib.fixture_dir("cvrplib").glob("*.vrp"))
    if source == "jsplib":
        folder = jsplib.fixture_dir()
        if not folder.exists():
            return []
        return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix == "")
    return []
