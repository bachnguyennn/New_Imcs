# Traffic Signal Controller (Modified Webster) – Overview

This document explains the code changes that added a signal-planning tool to the project, what the controller does, and what outputs it produces in both code and the Streamlit app.

## What changed in the repo

- Added `traffic_signal.py` with `ModifiedWebsterRoundabout`:
  - A small controller that computes a cycle length and green-time allocation using a Webster-style formula adapted for roundabouts.
- Updated `streamlit_app.py`:
  - New sidebar section “Traffic signal (Modified Webster)” with parameters and a "Compute signal plan" button.
  - Results display in the main pane (cycle length metric, table of green seconds, and a bar chart).
  - Minor cleanups: lazy-import of the simulator for faster app load, clearer field names, and an Arm legend (Arm0→N, Arm1→E, Arm2→S, Arm3→W).

## What the controller does

The controller implements a Modified Webster method to propose a fixed-time signal plan for the roundabout approaches. It follows the standard Webster approach with a small adaptation factor suitable for compact roundabouts.

High-level steps:
1. Compute critical flow ratios for each phase (approach) \(y_i = \text{flow}_i / s\), where `s` is saturation flow (veh/hr).
2. Sum ratios \(Y = \sum y_i\).
3. Compute cycle length using Webster’s formula with a reduction factor:
   \[ C = \max(\text{min\_cycle}, \min(\text{max\_cycle}, ((1.5L + 5) / (1 - Y)) \times \text{roundabout\_factor})) \]
   where `L` is total lost time per cycle (≈ lost_time_per_phase × number_of_phases).
4. Allocate effective green to each phase proportionally to demand: \( g_i = (y_i / Y) * (C - L) \), enforcing a minimum green per phase and re-normalizing if needed.

This yields a single cycle length `C` and a green-time dictionary `{phase: seconds}`.

## Inputs and outputs (code contract)

Class: `ModifiedWebsterRoundabout`
- Constructor parameters:
  - `phases: List[str]` – ordered phase names (e.g., `["N","E","S","W"]`).
  - `saturation_flow: float = 1800.0` – vehicles/hour per approach.
  - `lost_time_per_phase: float = 4.0` – seconds of lost time per phase (start-up + clearance).
  - `roundabout_factor: float = 0.9` – reduction factor (0–1] to better fit roundabouts.
- Primary API: `get_signal_plan(flows: Dict[str, float], min_cycle=20, max_cycle=180, min_green=4)`
  - `flows` are approach demands in vehicles/hour keyed by phase (e.g., `{"N": 300, "E": 200, ...}`).
  - Returns `(cycle_length_seconds: float, greens: Dict[str, float])`.

Example (Python):
```python
from traffic_signal import ModifiedWebsterRoundabout
flows = {"N": 300, "E": 200, "S": 250, "W": 150}
ctrl = ModifiedWebsterRoundabout(phases=["N","E","S","W"]) 
cycle, greens = ctrl.get_signal_plan(flows)
print(cycle)
print(greens)
```

## How to use it in the Streamlit app

1. Start the app from the project root:
   ```bash
   streamlit run streamlit_app.py
   ```
2. In the sidebar, under “Demand”, set arrival rates λ for Arm0–Arm3 (veh/s). The Arm legend maps them to N/E/S/W.
3. In “Traffic signal (Modified Webster)”, optionally adjust:
   - Saturation flow (veh/hr),
   - Lost time per phase (s),
   - Roundabout factor (0.5–1.0).
4. Click “Compute signal plan”. The main pane shows:
   - Cycle length (seconds),
   - A table of green_s per approach (N/E/S/W) and the flows used (veh/hr),
   - A bar chart of green seconds.

Note: The signal planner is currently advisory — it does not yet change the microsimulation logic; it provides a plan you can evaluate alongside the sim outputs.

## Example output (typical)

For flows `N=300, E=200, S=250, W=150` veh/hr, default parameters often yield something like:
- `cycle ≈ 52.2 s`
- `greens ≈ {"N": 12.07, "E": 8.04, "S": 10.06, "W": 6.03}`

Exact values will depend on your chosen parameters (saturation flow, lost times, reduction factor, min/max cycle, min green).

## Tuning tips
- If `Y = sum(flow_i/saturation_flow)` gets close to 1.0, the cycle can grow large; adjust `saturation_flow` or `lost_time_per_phase`, or cap with `max_cycle`.
- `roundabout_factor` < 1.0 slightly shortens Webster’s cycle — a pragmatic tweak for compact roundabouts.
- Use `min_green` to guarantee a minimum service time for minor approaches; the allocator re-normalizes if mins exceed the available green.

## Limitations & next steps
- The current signal plan is static (fixed-time) and not yet integrated with the microsim as an active control.
- Potential extensions:
  - Apply the plan dynamically inside the simulation loop.
  - Add pedestrian phases and intergreen checks.
  - Support multi-ring or coordinated corridors.
