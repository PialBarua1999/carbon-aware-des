from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import math
import random


@dataclass
class Job:
    job_id: int
    arrival_time: int          # Time step (e.g., minutes) when job arrives
    duration: int              # Required execution time steps
    deadline: int              # Latest allowed completion time step (SLA)
    gpu_power_watts: float     # Average GPU power draw during execution (W)
    remaining_time: int = field(init=False)
    start_time: int | None = None
    completion_time: int | None = None
    decision_reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.remaining_time = self.duration


def generate_carbon_profile(num_steps: int = 120, seed: int = 42) -> List[float]:
    """
    Generate a realistic diurnal carbon intensity profile (gCO2 / kWh)
    with a daily wave plus minor grid fluctuations.
    """
    rng = random.Random(seed)
    profile: List[float] = []
    for t in range(num_steps):
        # Smooth daily cycle + secondary wave
        diurnal = 360.0 + 140.0 * math.sin(2.0 * math.pi * (t - 24) / 96.0)
        ripple = 25.0 * math.cos(2.0 * math.pi * t / 18.0)
        noise = rng.uniform(-12.0, 12.0)
        intensity = max(120.0, diurnal + ripple + noise)
        profile.append(round(intensity, 2))
    return profile


def generate_jobs(num_steps: int = 90, seed: int = 101) -> List[Job]:
    """Generate a workload of GPU inference/batch jobs with deadlines (SLAs)."""
    rng = random.Random(seed)
    jobs: List[Job] = []
    job_id = 1
    for t in range(0, num_steps, 3):
        if rng.random() < 0.75:
            duration = rng.randint(1, 3)
            # Slack allows deferring when grid carbon is high, while keeping a real SLA
            slack = rng.randint(2, 14)
            deadline = t + duration + slack
            power_watts = rng.choice([180.0, 220.0, 260.0, 300.0])
            jobs.append(
                Job(
                    job_id=job_id,
                    arrival_time=t,
                    duration=duration,
                    deadline=deadline,
                    gpu_power_watts=power_watts,
                )
            )
            job_id += 1
    return jobs


def compute_step_carbon_grams(power_watts: float, step_minutes: float, carbon_intensity_g_per_kwh: float) -> float:
    """
    Energy (kWh) = Power (W) * Time (hours) / 1000
    Carbon (gCO2) = Energy (kWh) * Carbon Intensity (gCO2 / kWh)
    """
    energy_kwh = (power_watts * (step_minutes / 60.0)) / 1000.0
    return energy_kwh * carbon_intensity_g_per_kwh


def clone_jobs(jobs: List[Job]) -> List[Job]:
    return [
        Job(
            job_id=j.job_id,
            arrival_time=j.arrival_time,
            duration=j.duration,
            deadline=j.deadline,
            gpu_power_watts=j.gpu_power_watts,
        )
        for j in jobs
    ]


def simulate_baseline(
    jobs_template: List[Job],
    carbon_profile: List[float],
    step_minutes: float = 5.0,
) -> Tuple[dict, List[Job]]:
    """
    Baseline policy: Run jobs as soon as the GPU is free (FIFO),
    ignoring grid carbon intensity.
    """
    jobs = clone_jobs(jobs_template)
    pending: List[Job] = []
    active_job: Job | None = None
    total_carbon_g = 0.0
    busy_steps = 0

    max_step = len(carbon_profile)
    for t in range(max_step):
        for j in jobs:
            if j.arrival_time == t:
                pending.append(j)

        if active_job is None and pending:
            # Earliest arrival (FIFO)
            pending.sort(key=lambda x: (x.arrival_time, x.job_id))
            active_job = pending.pop(0)
            if active_job.start_time is None:
                active_job.start_time = t

        if active_job is not None:
            carbon_g = compute_step_carbon_grams(
                active_job.gpu_power_watts, step_minutes, carbon_profile[t]
            )
            total_carbon_g += carbon_g
            busy_steps += 1
            active_job.remaining_time -= 1
            active_job.decision_reasons.append("baseline_immediate")

            if active_job.remaining_time == 0:
                active_job.completion_time = t + 1
                active_job = None

    # Flush remaining pending jobs past simulation horizon if any
    t = max_step
    while active_job is not None or pending:
        if active_job is None and pending:
            pending.sort(key=lambda x: (x.arrival_time, x.job_id))
            active_job = pending.pop(0)
            if active_job.start_time is None:
                active_job.start_time = t
        if active_job is not None:
            intensity = carbon_profile[-1]
            total_carbon_g += compute_step_carbon_grams(
                active_job.gpu_power_watts, step_minutes, intensity
            )
            busy_steps += 1
            active_job.remaining_time -= 1
            if active_job.remaining_time == 0:
                active_job.completion_time = t + 1
                active_job = None
        t += 1

    sla_misses = sum(
        1 for j in jobs if j.completion_time is None or j.completion_time > j.deadline
    )
    avg_wait = sum(
        (j.start_time - j.arrival_time) for j in jobs if j.start_time is not None
    ) / max(1, len(jobs))

    metrics = {
        "policy": "Baseline (Immediate FIFO)",
        "total_carbon_g": total_carbon_g,
        "sla_misses": sla_misses,
        "total_jobs": len(jobs),
        "avg_wait_steps": avg_wait,
        "gpu_busy_steps": busy_steps,
    }
    return metrics, jobs


def simulate_carbon_aware(
    jobs_template: List[Job],
    carbon_profile: List[float],
    carbon_threshold: float,
    step_minutes: float = 5.0,
) -> Tuple[dict, List[Job]]:
    """
    Carbon-aware policy with SLA deadline guard:
    - If a job's remaining slack cannot survive waiting another step without risking
      missing its deadline, run it immediately (SLA guard).
    - Otherwise, run only when current carbon intensity <= carbon_threshold.
    - If multiple jobs are eligible, prioritize tightest deadline first (EDF).
    """
    jobs = clone_jobs(jobs_template)
    pending: List[Job] = []
    active_job: Job | None = None
    total_carbon_g = 0.0
    busy_steps = 0
    sla_guard_runs = 0
    deferred_steps = 0

    max_step = len(carbon_profile)
    for t in range(max_step):
        for j in jobs:
            if j.arrival_time == t:
                pending.append(j)

        if active_job is None and pending:
            # Sort by urgency (tightest deadline first)
            pending.sort(key=lambda x: (x.deadline, x.arrival_time, x.job_id))
            current_carbon = carbon_profile[t]

            chosen_idx: int | None = None
            chosen_reason = ""

            for idx, candidate in enumerate(pending):
                # Time remaining until deadline if we start right now:
                # Must finish by `candidate.deadline`, taking `candidate.remaining_time` steps.
                latest_safe_start = candidate.deadline - candidate.remaining_time
                must_run_for_sla = t >= latest_safe_start
                is_low_carbon = current_carbon <= carbon_threshold

                if must_run_for_sla:
                    chosen_idx = idx
                    chosen_reason = "sla_deadline_guard"
                    sla_guard_runs += 1
                    break
                if is_low_carbon:
                    chosen_idx = idx
                    chosen_reason = "low_carbon_window"
                    break

            if chosen_idx is not None:
                active_job = pending.pop(chosen_idx)
                if active_job.start_time is None:
                    active_job.start_time = t
                active_job.decision_reasons.append(chosen_reason)
            else:
                deferred_steps += 1

        if active_job is not None:
            carbon_g = compute_step_carbon_grams(
                active_job.gpu_power_watts, step_minutes, carbon_profile[t]
            )
            total_carbon_g += carbon_g
            busy_steps += 1
            active_job.remaining_time -= 1

            if active_job.remaining_time == 0:
                active_job.completion_time = t + 1
                active_job = None

    # Flush any remaining jobs if simulation horizon ends while jobs are queued
    t = max_step
    while active_job is not None or pending:
        if active_job is None and pending:
            pending.sort(key=lambda x: (x.deadline, x.arrival_time, x.job_id))
            active_job = pending.pop(0)
            if active_job.start_time is None:
                active_job.start_time = t
        if active_job is not None:
            intensity = carbon_profile[-1]
            total_carbon_g += compute_step_carbon_grams(
                active_job.gpu_power_watts, step_minutes, intensity
            )
            busy_steps += 1
            active_job.remaining_time -= 1
            if active_job.remaining_time == 0:
                active_job.completion_time = t + 1
                active_job = None
        t += 1

    sla_misses = sum(
        1 for j in jobs if j.completion_time is None or j.completion_time > j.deadline
    )
    avg_wait = sum(
        (j.start_time - j.arrival_time) for j in jobs if j.start_time is not None
    ) / max(1, len(jobs))

    metrics = {
        "policy": f"Carbon-Aware + SLA Guard (threshold={carbon_threshold:.1f} gCO2/kWh)",
        "total_carbon_g": total_carbon_g,
        "sla_misses": sla_misses,
        "total_jobs": len(jobs),
        "avg_wait_steps": avg_wait,
        "gpu_busy_steps": busy_steps,
        "sla_guard_dispatches": sla_guard_runs,
        "idle_deferred_steps": deferred_steps,
    }
    return metrics, jobs


def main() -> None:
    step_minutes = 5.0
    carbon_profile = generate_carbon_profile(num_steps=140, seed=42)
    jobs = generate_jobs(num_steps=95, seed=101)

    # Set threshold around the 45th percentile of the observed carbon profile
    sorted_carbon = sorted(carbon_profile)
    threshold = sorted_carbon[int(0.45 * len(sorted_carbon))]

    base_metrics, _ = simulate_baseline(jobs, carbon_profile, step_minutes=step_minutes)
    aware_metrics, aware_jobs = simulate_carbon_aware(
        jobs, carbon_profile, carbon_threshold=threshold, step_minutes=step_minutes
    )

    carbon_saved_g = base_metrics["total_carbon_g"] - aware_metrics["total_carbon_g"]
    carbon_saved_pct = (carbon_saved_g / base_metrics["total_carbon_g"]) * 100.0

    print("=== GPU Carbon-Aware Scheduling Simulation ===")
    print(f"Total Jobs: {len(jobs)} | Step Size: {step_minutes} min")
    print(
        f"Carbon Intensity Range: {min(carbon_profile):.1f} - {max(carbon_profile):.1f} gCO2/kWh "
        f"(Threshold: {threshold:.1f} gCO2/kWh)\n"
    )

    for m in (base_metrics, aware_metrics):
        print(f"[{m['policy']}]")
        print(f"  Total Carbon Emitted : {m['total_carbon_g']:.2f} gCO2")
        print(f"  SLA Deadline Misses  : {m['sla_misses']} / {m['total_jobs']}")
        print(f"  Avg Wait Time        : {m['avg_wait_steps']:.2f} steps")
        print(f"  GPU Active Steps     : {m['gpu_busy_steps']}")
        if "sla_guard_dispatches" in m:
            print(f"  SLA Guard Dispatches : {m['sla_guard_dispatches']}")
            print(f"  Deferred Idle Steps  : {m['idle_deferred_steps']}")
        print()

    print(f"Carbon Saved: {carbon_saved_g:.2f} gCO2 ({carbon_saved_pct:.2f}%)")

    # Show a few sample job traces from the carbon-aware run
    print("\nSample Job Traces (first 6 jobs):")
    for j in aware_jobs[:6]:
        sla_ok = j.completion_time is not None and j.completion_time <= j.deadline
        print(
            f"  Job {j.job_id:02d}: arrive={j.arrival_time:02d}, start={j.start_time}, "
            f"done={j.completion_time}, deadline={j.deadline}, sla_met={sla_ok}, "
            f"reasons={j.decision_reasons}"
        )


if __name__ == "__main__":
    main()