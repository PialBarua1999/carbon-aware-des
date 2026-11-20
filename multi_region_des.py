
import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RegionConfig:
    name: str
    base_carbon: float          # Mean carbon intensity (gCO2/kWh)
    amplitude: float            # Swing amplitude across the day (gCO2/kWh)
    peak_hour: float            # Hour of day (0-24) when carbon peaks
    num_servers: int            # Available compute slots in this region
    carbon_threshold: float     # Preferred max carbon intensity (gCO2/kWh)
    routing_delay_hr: float = 0.05  # Network/transfer overhead if routed remotely


@dataclass
class Job:
    job_id: int
    home_region: str
    arrival_time: float         # Hours
    service_time: float         # Hours
    sla_max_latency: float      # Hours
    power_kw: float = 0.5       # Average power draw (kW)
    assigned_region: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    carbon_emitted_g: float = 0.0
    sla_met: Optional[bool] = None
    routed_remotely: bool = False

    @property
    def deadline(self) -> float:
        return self.arrival_time + self.sla_max_latency


class GridCarbonModel:
    """Deterministic multi-region carbon intensity model (gCO2/kWh)."""

    def __init__(self, regions: Dict[str, RegionConfig]):
        self.regions = regions

    def intensity(self, region_name: str, t_hours: float) -> float:
        cfg = self.regions[region_name]
        # Peak at `peak_hour`, trough 12 hours later
        diurnal = cfg.base_carbon + cfg.amplitude * math.cos(
            2.0 * math.pi * (t_hours - cfg.peak_hour) / 24.0
        )
        # Small deterministic secondary harmonic for realistic curve shape
        ripple = 12.0 * math.sin(2.0 * math.pi * t_hours / 8.0 + len(region_name))
        return max(50.0, diurnal + ripple)


class MultiRegionDES:
    def __init__(
        self,
        policy: str,
        regions: Dict[str, RegionConfig],
        sim_duration_hours: float = 24.0,
        arrival_rate_per_hour: float = 6.0,
        mean_service_hours: float = 0.35,
        default_sla_hours: float = 1.4,
        seed: int = 42,
    ):
        if policy not in {"FIFO", "EDF", "CARBON_AWARE"}:
            raise ValueError("Policy must be one of: FIFO, EDF, CARBON_AWARE")

        self.policy = policy
        self.regions = regions
        self.carbon_model = GridCarbonModel(regions)
        self.sim_duration_hours = sim_duration_hours
        self.arrival_rate_per_hour = arrival_rate_per_hour
        self.mean_service_hours = mean_service_hours
        self.default_sla_hours = default_sla_hours

        self.rng = random.Random(seed)
        self.now: float = 0.0
        self.event_seq: int = 0
        self.event_heap: List[Tuple[float, int, str, Optional[Job]]] = []

        # Per-region state
        self.busy_servers: Dict[str, int] = {r: 0 for r in regions}
        self.queues: Dict[str, List[Job]] = {r: [] for r in regions}
        self.completed_jobs: List[Job] = []
        self.deferral_events: int = 0
        self.remote_routes: int = 0

    def _schedule_event(self, event_time: float, event_type: str, job: Optional[Job] = None) -> None:
        if event_time <= self.sim_duration_hours + 1e-9:
            self.event_seq += 1
            heapq.heappush(self.event_heap, (event_time, self.event_seq, event_type, job))

    def _generate_arrivals(self) -> None:
        region_names = list(self.regions.keys())
        t = 0.0
        job_id = 1
        while True:
            t += self.rng.expovariate(self.arrival_rate_per_hour)
            if t > self.sim_duration_hours:
                break

            home_region = self.rng.choice(region_names)
            service_time = max(0.05, self.rng.expovariate(1.0 / self.mean_service_hours))
            sla_latency = max(
                service_time * 1.25,
                self.rng.gauss(self.default_sla_hours, 0.25),
            )
            job = Job(
                job_id=job_id,
                home_region=home_region,
                arrival_time=t,
                service_time=service_time,
                sla_max_latency=sla_latency,
            )
            self._schedule_event(t, "ARRIVAL", job)
            job_id += 1

    def _estimate_start_time(self, region_name: str, job: Job) -> float:
        """Rough earliest start time in `region_name` considering transfer delay and queue."""
        transfer = 0.0 if region_name == job.home_region else self.regions[region_name].routing_delay_hr
        earliest = self.now + transfer
        if self.busy_servers[region_name] < self.regions[region_name].num_servers:
            return earliest
        # Approximate wait by summing queued service times divided by servers
        queued_work = sum(q.service_time for q in self.queues[region_name])
        est_wait = queued_work / max(1, self.regions[region_name].num_servers)
        return earliest + est_wait * 0.5

    def _choose_target_region(self, job: Job) -> str:
        """Selects the best region for an arriving job under the current policy."""
        home = job.home_region

        if self.policy == "FIFO":
            # Prefer home region if a server is free; otherwise least loaded queue
            if self.busy_servers[home] < self.regions[home].num_servers and not self.queues[home]:
                return home
            return min(
                self.regions.keys(),
                key=lambda r: (len(self.queues[r]) + self.busy_servers[r] / self.regions[r].num_servers),
            )

        if self.policy == "EDF":
            # Route to the region giving the earliest projected completion time
            return min(
                self.regions.keys(),
                key=lambda r: self._estimate_start_time(r, job) + job.service_time,
            )

        # CARBON_AWARE routing:
        # Score each region by expected carbon emission, penalizing regions that would miss SLA
        best_region = home
        best_score = float("inf")

        for r_name, cfg in self.regions.items():
            est_start = self._estimate_start_time(r_name, job)
            est_finish = est_start + job.service_time
            carbon_rate = self.carbon_model.intensity(r_name, est_start)
            est_carbon = job.power_kw * job.service_time * carbon_rate

            # Add large penalty if projected completion misses SLA deadline
            sla_penalty = 1e6 if est_finish > job.deadline else 0.0
            # Slight tie-breaker preference for local/clean regions
            threshold_bonus = -5.0 if carbon_rate <= cfg.carbon_threshold else 0.0
            score = est_carbon + sla_penalty + threshold_bonus

            if score < best_score:
                best_score = score
                best_region = r_name

        return best_region

    def _route_arrival(self, job: Job) -> None:
        target = self._choose_target_region(job)
        job.assigned_region = target
        job.routed_remotely = (target != job.home_region)
        if job.routed_remotely:
            self.remote_routes += 1

        delay = self.regions[target].routing_delay_hr if job.routed_remotely else 0.0
        self.queues[target].append(job)
        self._schedule_event(self.now + delay, "DISPATCH_CHECK", None)

    def _pick_job_from_region(self, region_name: str) -> Optional[Job]:
        queue = self.queues[region_name]
        if not queue:
            return None

        cfg = self.regions[region_name]
        carbon_now = self.carbon_model.intensity(region_name, self.now)

        if self.policy == "FIFO":
            queue.sort(key=lambda j: j.arrival_time)
            return queue.pop(0)

        if self.policy == "EDF":
            queue.sort(key=lambda j: (j.deadline, j.arrival_time))
            return queue.pop(0)

        # CARBON_AWARE local dispatch:
        # 1. Run urgent jobs immediately to protect SLA
        urgent = [j for j in queue if (self.now + j.service_time) >= (j.deadline - 0.05)]
        if urgent:
            urgent.sort(key=lambda j: j.deadline)
            chosen = urgent[0]
            queue.remove(chosen)
            return chosen

        # 2. If local grid carbon is within threshold, run earliest-deadline job
        if carbon_now <= cfg.carbon_threshold:
            queue.sort(key=lambda j: (j.deadline, j.arrival_time))
            return queue.pop(0)

        # 3. Otherwise defer non-urgent jobs until carbon drops or deadline approaches
        self.deferral_events += 1
        return None

    def _dispatch_region(self, region_name: str) -> None:
        cfg = self.regions[region_name]
        while self.busy_servers[region_name] < cfg.num_servers and self.queues[region_name]:
            job = self._pick_job_from_region(region_name)
            if job is None:
                break

            self.busy_servers[region_name] += 1
            job.start_time = self.now
            finish_time = self.now + job.service_time
            job.completion_time = finish_time

            carbon_rate = self.carbon_model.intensity(region_name, self.now)
            job.carbon_emitted_g = job.power_kw * job.service_time * carbon_rate
            job.sla_met = (finish_time <= job.deadline + 1e-9)

            self._schedule_event(finish_time, "COMPLETION", job)

    def _dispatch_all_regions(self) -> None:
        for r_name in self.regions:
            self._dispatch_region(r_name)

    def run(self) -> Dict[str, float]:
        self._generate_arrivals()

        # Periodic check every 15 minutes (0.25 hr) for carbon-aware deferral release
        tick = 0.0
        while tick <= self.sim_duration_hours:
            self._schedule_event(tick, "POLICY_TICK", None)
            tick += 0.25

        while self.event_heap:
            event_time, _, event_type, job = heapq.heappop(self.event_heap)
            self.now = event_time

            if event_type == "ARRIVAL" and job is not None:
                self._route_arrival(job)
                self._dispatch_all_regions()

            elif event_type == "DISPATCH_CHECK":
                self._dispatch_all_regions()

            elif event_type == "COMPLETION" and job is not None:
                r_name = job.assigned_region or job.home_region
                self.busy_servers[r_name] = max(0, self.busy_servers[r_name] - 1)
                self.completed_jobs.append(job)
                self._dispatch_region(r_name)

            elif event_type == "POLICY_TICK":
                self._dispatch_all_regions()

        total = len(self.completed_jobs)
        sla_met = sum(1 for j in self.completed_jobs if j.sla_met)
        total_carbon = sum(j.carbon_emitted_g for j in self.completed_jobs)
        avg_lat_hr = (
            sum((j.completion_time - j.arrival_time) for j in self.completed_jobs if j.completion_time) / total
            if total
            else 0.0
        )

        return {
            "policy": self.policy,
            "completed": total,
            "sla_pct": (sla_met / total * 100.0) if total else 0.0,
            "avg_lat_min": avg_lat_hr * 60.0,
            "carbon_g": total_carbon,
            "remote_routes": self.remote_routes,
            "defers": self.deferral_events,
        }


def main() -> None:
    # Three regions with staggered clean-energy peaks and different capacities
    regions = {
        "us-west-solar": RegionConfig(
            name="us-west-solar",
            base_carbon=260.0,
            amplitude=150.0,
            peak_hour=21.0,       # Cleanest around 09:00, dirtiest at 21:00
            num_servers=2,
            carbon_threshold=260.0,
            routing_delay_hr=0.06,
        ),
        "eu-central-wind": RegionConfig(
            name="eu-central-wind",
            base_carbon=340.0,
            amplitude=120.0,
            peak_hour=14.0,       # Cleanest overnight/early morning
            num_servers=2,
            carbon_threshold=320.0,
            routing_delay_hr=0.05,
        ),
        "ap-south-grid": RegionConfig(
            name="ap-south-grid",
            base_carbon=480.0,
            amplitude=90.0,
            peak_hour=19.0,       # Evening peak demand
            num_servers=2,
            carbon_threshold=430.0,
            routing_delay_hr=0.08,
        ),
    }

    rows = []
    for policy in ["FIFO", "EDF", "CARBON_AWARE"]:
        sim = MultiRegionDES(
            policy=policy,
            regions=regions,
            sim_duration_hours=24.0,
            arrival_rate_per_hour=6.5,
            mean_service_hours=0.35,
            default_sla_hours=1.35,
            seed=2026,
        )
        rows.append(sim.run())

    header = (
        f"{'Policy':<14} | {'Jobs':<5} | {'SLA %':<6} | "
        f"{'Avg Lat (m)':<11} | {'Carbon (g)':<10} | {'Remote':<6} | {'Defers':<6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['policy']:<14} | {r['completed']:<5} | {r['sla_pct']:<6.1f} | "
            f"{r['avg_lat_min']:<11.1f} | {r['carbon_g']:<10.2f} | "
            f"{r['remote_routes']:<6} | {r['defers']:<6}"
        )


if __name__ == "__main__":
    main()