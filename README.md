# Carbon-Aware Discrete-Event Simulator (DES)
...
A discrete-event simulation framework in Python for evaluating **FIFO**, **Earliest Deadline First (EDF)**, and **Carbon-Aware Deferral / Multi-Region Routing** scheduling policies under synthetic grid carbon intensity curves, Poisson job arrivals, and Service Level Agreement (SLA) latency constraints.

---

## Key Features

* **Event-Driven Simulation (`heapq`):** Jumps directly across discrete events (`ARRIVAL`, `COMPLETION`, `DISPATCH_CHECK`, `POLICY_TICK`) for fast and reproducible execution.
* **Synthetic Grid Carbon Intensity ($g\text{CO}_2/\text{kWh}$):** Models time-varying clean/high-carbon grid profiles (lower carbon during peak renewable generation hours, higher carbon during peak demand hours).
* **SLA Latency Enforcement:** Each job defines an `sla_max_latency` (`deadline = arrival_time + sla_max_latency`) to measure SLA compliance and avoid late execution.
* **Carbon-Aware Deferral & Multi-Region Routing:**
  * Defers non-urgent jobs when carbon intensity exceeds a threshold, while releasing urgent jobs before they breach their SLA deadline.
  * Routes jobs across regions (`multi_region_des.py`) with region-specific carbon curves, server capacities, and cross-region transfer delays.

---

## Project Structure

```text
carbon-aware-des/
├── .gitignore
├── README.md
├── carbon_des.py          # Single-region discrete-event simulator
└── multi_region_des.py    # Multi-region simulator with job routing
```

---

## Prerequisites & Running the Simulators

* **Python 3.9+** (uses Python standard library only; no external dependencies required).

Run the single-region simulator:
```bash
python carbon_des.py
```

Run the multi-region simulator:
```bash
python multi_region_des.py
```

---

## Scheduling Policies Compared

| Policy | How It Works | Trade-offs |
| :--- | :--- | :--- |
| **FIFO** | Executes jobs in arrival order (and keeps jobs local when capacity allows). | Simple baseline; ignores deadlines and grid carbon emissions. |
| **EDF** | Prioritizes jobs with the earliest SLA deadline (and routes for earliest completion). | Strong SLA compliance; may execute during high-carbon periods. |
| **CARBON_AWARE** | Defers non-urgent jobs above carbon threshold and routes to lower-carbon regions within SLA limits. | Reduces total carbon emissions while protecting urgent SLA deadlines. |

---

## বাংলা সারসংক্ষেপ (Bengali Overview)

এই রিপোজিটরিতে সিন্থেটিক কার্বন ইনটেনসিটি ডেটা, জব এরাইভাল রেট এবং SLA ল্যাটেন্সি কনস্ট্রেইন্টের ভিত্তিতে **FIFO**, **EDF** এবং **Carbon-Aware** শিডিউলিং পলিসি যাচাই করার জন্য দুটি Python Discrete-Event Simulator দেওয়া হয়েছে:

* **`carbon_des.py`:** একক রিজিয়নের সার্ভার, SLA ডেডলাইন এবং কার্বন থ্রেশহোল্ড অনুযায়ী জব defer করার সিমুলেশন।
* **`multi_region_des.py`:** একাধিক রিজিয়নের আলাদা কার্বন কার্ভ, সার্ভার ক্যাপাসিটি এবং ক্রস-রিজিয়ন রাউটিং ডিলেসহ সিমুলেশন।

### রান করার কমান্ড (বাংলা)
```bash
python carbon_des.py
python multi_region_des.py