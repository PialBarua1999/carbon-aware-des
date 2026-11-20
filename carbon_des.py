import heapq
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Job:
    job_id: int
    arrival_time: float       # ঘণ্টায় (hours)
    service_time: float       # কাজের প্রয়োজনীয় সময় (hours)
    sla_max_latency: float    # সর্বোচ্চ অনুমোদিত মোট ল্যাটেন্সি (hours)
    power_kw: float = 0.5     # জব চলাকালীন গড় সার্ভার পাওয়ার (kW)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    carbon_emitted_g: float = 0.0
    sla_met: Optional[bool] = None

    @property
    def deadline(self) -> float:
        return self.arrival_time + self.sla_max_latency


def synthetic_carbon_intensity(t_hours: float, rng: random.Random) -> float:
    """
    সিন্থেটিক কার্বন ইনটেনসিটি (gCO2/kWh):
    দুপুর ১২টার দিকে (t % 24 == 12) গ্রিডে ক্লিন এনার্জি বেশি থাকায় কার্বন কম থাকে,
    আর সন্ধ্যা/রাতে কার্বন ইনটেনসিটি তুলনামূলক বেশি থাকে।
    """
    # দুপুর ১২টায় মিনিমাম (~180 gCO2/kWh) এবং মধ্যরাতে ম্যাক্সিমাম (~520 gCO2/kWh)
    diurnal = 350.0 + 150.0 * math.cos(2.0 * math.pi * (t_hours - 12.0) / 24.0)
    noise = rng.gauss(0.0, 18.0)
    return max(80.0, diurnal + noise)


class DiscreteEventSimulator:
    def __init__(
        self,
        policy: str,
        sim_duration_hours: float = 24.0,
        num_servers: int = 2,
        arrival_rate_per_hour: float = 4.0,
        mean_service_hours: float = 0.35,
        default_sla_hours: float = 1.2,
        carbon_threshold: float = 340.0,
        seed: int = 42,
    ):
        if policy not in {"FIFO", "EDF", "CARBON_AWARE"}:
            raise ValueError("Policy must be one of: FIFO, EDF, CARBON_AWARE")
        self.policy = policy
        self.sim_duration_hours = sim_duration_hours
        self.num_servers = num_servers
        self.arrival_rate_per_hour = arrival_rate_per_hour
        self.mean_service_hours = mean_service_hours
        self.default_sla_hours = default_sla_hours
        self.carbon_threshold = carbon_threshold

        self.rng = random.Random(seed)
        self.now: float = 0.0
        self.event_seq: int = 0
        self.event_heap: List[Tuple[float, int, str, Optional[Job]]] = []

        self.busy_servers: int = 0
        self.waiting_queue: List[Job] = []
        self.completed_jobs: List[Job] = []
        self.deferred_count: int = 0

    def _schedule_event(self, event_time: float, event_type: str, job: Optional[Job] = None) -> None:
        if event_time <= self.sim_duration_hours + 1e-9:
            self.event_seq += 1
            heapq.heappush(self.event_heap, (event_time, self.event_seq, event_type, job))

    def _generate_arrivals(self) -> None:
        """Poisson arrival process: Exponential inter-arrival times"""
        t = 0.0
        job_id = 1
        while True:
            inter_arrival = self.rng.expovariate(self.arrival_rate_per_hour)
            t += inter_arrival
            if t > self.sim_duration_hours:
                break

            service_time = max(0.05, self.rng.expovariate(1.0 / self.mean_service_hours))
            # কিছু জবের SLA টাইট এবং কিছু জবের কিছুটা ফ্লেক্সিবল রাখা হয়েছে
            sla_latency = max(
                service_time * 1.2,
                self.rng.gauss(self.default_sla_hours, 0.25),
            )
            job = Job(
                job_id=job_id,
                arrival_time=t,
                service_time=service_time,
                sla_max_latency=sla_latency,
            )
            self._schedule_event(t, "ARRIVAL", job)
            job_id += 1

    def _pick_next_job(self) -> Optional[Job]:
        """পলিসি অনুযায়ী পরবর্তী চালানোর উপযোগী জব নির্বাচন করে"""
        if not self.waiting_queue:
            return None

        current_carbon = synthetic_carbon_intensity(self.now, self.rng)

        if self.policy == "FIFO":
            self.waiting_queue.sort(key=lambda j: j.arrival_time)
            return self.waiting_queue.pop(0)

        if self.policy == "EDF":
            self.waiting_queue.sort(key=lambda j: (j.deadline, j.arrival_time))
            return self.waiting_queue.pop(0)

        # CARBON_AWARE নীতি:
        # ১) আগে জরুরি জবগুলো চেক করি (এখন শুরু না করলে SLA deadline মিস হবে)
        urgent_jobs = [
            j for j in self.waiting_queue
            if (self.now + j.service_time) >= (j.deadline - 0.05)
        ]
        if urgent_jobs:
            urgent_jobs.sort(key=lambda j: j.deadline)
            chosen = urgent_jobs[0]
            self.waiting_queue.remove(chosen)
            return chosen

        # ২) কার্বন ইনটেনসিটি থ্রেশহোল্ডের নিচে থাকলে earliest-deadline জব চালাই
        if current_carbon <= self.carbon_threshold:
            self.waiting_queue.sort(key=lambda j: (j.deadline, j.arrival_time))
            return self.waiting_queue.pop(0)

        # ৩) কার্বন বেশি এবং জবটি এখনো জরুরি নয় -> Defer করি
        self.deferred_count += 1
        return None

    def _dispatch_jobs(self) -> None:
        while self.busy_servers < self.num_servers and self.waiting_queue:
            job = self._pick_next_job()
            if job is None:
                break

            self.busy_servers += 1
            job.start_time = self.now
            finish_time = self.now + job.service_time
            job.completion_time = finish_time

            # কার্বন নিঃসরণ হিসাব: Power (kW) * Time (h) * Carbon Intensity (gCO2/kWh)
            carbon_now = synthetic_carbon_intensity(self.now, self.rng)
            job.carbon_emitted_g = job.power_kw * job.service_time * carbon_now
            job.sla_met = (finish_time <= job.deadline + 1e-9)

            self._schedule_event(finish_time, "COMPLETION", job)

    def run(self) -> Dict[str, float]:
        self._generate_arrivals()

        # Carbon-aware পলিসির জন্য প্রতি ১৫ মিনিটে (0.25 hr) কার্বন ও deferred জব পুনর্মূল্যায়ন
        tick = 0.0
        while tick <= self.sim_duration_hours:
            self._schedule_event(tick, "POLICY_TICK", None)
            tick += 0.25

        while self.event_heap:
            event_time, _, event_type, job = heapq.heappop(self.event_heap)
            self.now = event_time

            if event_type == "ARRIVAL" and job is not None:
                self.waiting_queue.append(job)
                self._dispatch_jobs()

            elif event_type == "COMPLETION" and job is not None:
                self.busy_servers = max(0, self.busy_servers - 1)
                self.completed_jobs.append(job)
                self._dispatch_jobs()

            elif event_type == "POLICY_TICK":
                self._dispatch_jobs()

        total_completed = len(self.completed_jobs)
        sla_met_count = sum(1 for j in self.completed_jobs if j.sla_met)
        total_carbon_g = sum(j.carbon_emitted_g for j in self.completed_jobs)
        avg_latency_hr = (
            sum((j.completion_time - j.arrival_time) for j in self.completed_jobs if j.completion_time)
            / total_completed
            if total_completed > 0
            else 0.0
        )

        return {
            "policy": self.policy,
            "completed_jobs": total_completed,
            "sla_compliance_pct": (sla_met_count / total_completed * 100.0) if total_completed else 0.0,
            "avg_latency_min": avg_latency_hr * 60.0,
            "total_carbon_g": total_carbon_g,
            "deferral_events": self.deferred_count,
        }


def main() -> None:
    policies = ["FIFO", "EDF", "CARBON_AWARE"]
    results = []

    for p in policies:
        sim = DiscreteEventSimulator(
            policy=p,
            sim_duration_hours=24.0,
            num_servers=2,
            arrival_rate_per_hour=4.5,
            mean_service_hours=0.32,
            default_sla_hours=1.3,
            carbon_threshold=330.0,
            seed=101,
        )
        results.append(sim.run())

    print(f"{'Policy':<14} | {'Completed':<9} | {'SLA Met %':<9} | {'Avg Lat (m)':<11} | {'Carbon (g)':<10} | {'Defers':<6}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['policy']:<14} | "
            f"{r['completed_jobs']:<9} | "
            f"{r['sla_compliance_pct']:<9.1f} | "
            f"{r['avg_latency_min']:<11.1f} | "
            f"{r['total_carbon_g']:<10.2f} | "
            f"{r['deferral_events']:<6}"
        )


if __name__ == "__main__":
    main()