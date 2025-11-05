"""traffic_signal.py

Provides a ModifiedWebsterRoundabout class that computes a signal cycle and
green-time allocation using a Webster-based method adapted for roundabouts.

The implementation is intentionally small and self-contained. It computes:
- critical flow ratios y_i = flow_i / saturation_flow
- total ratio Y = sum(y_i)
- cycle length C using Webster: C = (1.5 * L + 5) / (1 - Y) where L is total lost time
  (with configurable reduction factor to adapt to roundabout characteristics)
- green times g_i = (y_i / Y) * (C - L) with min-green enforcement

Example:
    from traffic_signal import ModifiedWebsterRoundabout
    flows = {'N': 300, 'E': 200, 'S': 250, 'W': 150}
    ctrl = ModifiedWebsterRoundabout(phases=['N','E','S','W'])
    cycle, greens = ctrl.get_signal_plan(flows)
    print(cycle)
    print(greens)
"""

from typing import Dict, List, Tuple


class ModifiedWebsterRoundabout:
    """Controller using a Modified Webster method for roundabouts.

    Contract (simple):
    - Input: flows: Dict[phase_name, vehicles_per_hour]
    - Output: (cycle_length_seconds, Dict[phase_name, green_seconds])

    Parameters:
    - phases: ordered list of phases (strings)
    - saturation_flow: vehicles per hour per approach (default 1800)
    - lost_time_per_phase: seconds of lost time per phase (default 4)
    - roundabout_factor: multiplier (0 < f <= 1) to reduce Webster cycle for roundabout
      (default 0.9) — smaller values give slightly shorter cycles appropriate for compact
      multi-entry roundabouts.
    """

    def __init__(
        self,
        phases: List[str],
        saturation_flow: float = 1800.0,
        lost_time_per_phase: float = 4.0,
        roundabout_factor: float = 0.9,
    ) -> None:
        self.phases = list(phases)
        self.saturation_flow = float(saturation_flow)
        self.lost_time_per_phase = float(lost_time_per_phase)
        self.roundabout_factor = float(roundabout_factor)

    def _critical_flow_ratios(self, flows: Dict[str, float]) -> Dict[str, float]:
        """Return critical flow ratios y_i = flow_i / saturation_flow for each phase."""
        ratios = {}
        for p in self.phases:
            f = float(flows.get(p, 0.0))
            ratios[p] = f / self.saturation_flow
        return ratios

    def compute_cycle_length(
        self,
        flows: Dict[str, float],
        min_cycle: float = 20.0,
        max_cycle: float = 180.0,
    ) -> Tuple[float, float]:
        """Compute Webster-style cycle length.

        Returns (cycle_length_seconds, total_lost_time_seconds)
        """
        ratios = self._critical_flow_ratios(flows)
        Y = sum(ratios.values())
        n = len(self.phases)
        L = n * self.lost_time_per_phase

        if Y >= 1.0:
            # Oversaturated: return a large cycle but clamp to max_cycle
            C = max_cycle
        else:
            # Classic Webster: C0 = (1.5L + 5) / (1 - Y)
            C0 = (1.5 * L + 5.0) / (1.0 - Y)
            # Adjust for roundabout (empirical reduction factor)
            C = max(min_cycle, min(max_cycle, C0 * self.roundabout_factor))

        return float(C), float(L)

    def allocate_green_times(
        self,
        flows: Dict[str, float],
        cycle_length: float,
        lost_time: float,
        min_green: float = 4.0,
    ) -> Dict[str, float]:
        """Allocate green times to each phase using critical flow ratios.

        g_i = (y_i / Y) * (C - L) subject to min_green; if min_green constraints
        push allocation above available green, we clamp and re-normalize.
        """
        ratios = self._critical_flow_ratios(flows)
        Y = sum(ratios.values())
        available = cycle_length - lost_time
        if available <= 0:
            # No green time available: allocate minimums evenly
            return {p: min_green for p in self.phases}

        greens: Dict[str, float] = {}
        if Y <= 0:
            # No demand: split available green evenly
            per = available / max(1, len(self.phases))
            for p in self.phases:
                greens[p] = max(min_green, per)
            return greens

        # First pass: proportional by y_i
        for p in self.phases:
            y = ratios[p]
            g = (y / Y) * available
            greens[p] = max(min_green, g)

        # If sum exceeds available (because mins pushed it), trim proportionally
        total_assigned = sum(greens.values())
        if total_assigned > available:
            # scale down to available while keeping mins >= min_green
            # compute variable portion (assigned - min) and scale them
            mins = {p: min_green for p in self.phases}
            variable = {p: max(0.0, greens[p] - mins[p]) for p in self.phases}
            var_sum = sum(variable.values())
            if var_sum <= 1e-9:
                # all at min green, then we must compress mins (rare)
                factor = available / sum(mins.values())
                for p in self.phases:
                    greens[p] = max(0.5, mins[p] * factor)
            else:
                scale = max(0.0, (available - sum(mins.values()))) / var_sum
                for p in self.phases:
                    greens[p] = mins[p] + variable[p] * scale

        # Final check: ensure no negative and round to 2 decimals
        for p in self.phases:
            greens[p] = round(max(0.0, greens[p]), 2)

        return greens

    def get_signal_plan(
        self,
        flows: Dict[str, float],
        min_cycle: float = 20.0,
        max_cycle: float = 180.0,
        min_green: float = 4.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Return (cycle_length_seconds, greens dict) for the provided flows."""
        C, L = self.compute_cycle_length(flows, min_cycle=min_cycle, max_cycle=max_cycle)
        greens = self.allocate_green_times(flows, C, L, min_green=min_green)
        return C, greens


if __name__ == "__main__":
    # Quick manual test when run as a script
    sample_flows = {"N": 300, "E": 200, "S": 250, "W": 150}
    ctrl = ModifiedWebsterRoundabout(phases=["N", "E", "S", "W"])
    cycle, greens = ctrl.get_signal_plan(sample_flows)
    print("Cycle:", cycle)
    print("Greens:", greens)
