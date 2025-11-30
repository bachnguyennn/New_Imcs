
from typing import Dict, List, Tuple, Optional
import random
import argparse
from pathlib import Path
import pandas as pd


class ModifiedWebsterRoundabout:
    
    def __init__(
        self,
        phases: List[str],
        saturation_flow: float = 1800.0,
        lost_time_per_phase: float = 4.0,
        roundabout_factor: float = 0.9,
    ) -> None:
        if not phases:
            raise ValueError("phases must be a non-empty list of strings")
        if saturation_flow <= 0:
            raise ValueError("saturation_flow must be positive")
        if lost_time_per_phase < 0:
            raise ValueError("lost_time_per_phase must be non-negative")
        if not (0 < roundabout_factor <= 1.0):
            raise ValueError("roundabout_factor must be in (0, 1]")

        self.phases = list(phases)
        self.saturation_flow = float(saturation_flow)
        self.lost_time_per_phase = float(lost_time_per_phase)
        self.roundabout_factor = float(roundabout_factor)

    def _critical_flow_ratios(self, flows: Dict[str, float]) -> Dict[str, float]:
        """Return critical flow ratios y_i = flow_i / saturation_flow for each phase."""
        ratios = {}
        for p in self.phases:
            f = max(0.0, float(flows.get(p, 0.0)))
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
        if min_cycle <= 0 or max_cycle <= 0:
            raise ValueError("min_cycle and max_cycle must be positive")
        if min_cycle > max_cycle:
            raise ValueError("min_cycle cannot exceed max_cycle")
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

        # Ensure the cycle is never shorter than the total lost time.
        C = max(C, L)

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
        if min_green < 0:
            raise ValueError("min_green must be non-negative")
        ratios = self._critical_flow_ratios(flows)
        Y = sum(ratios.values())
        available = max(0.0, cycle_length - lost_time)
        required_min = min_green * len(self.phases)

        if available <= 1e-9:
            return {p: 0.0 for p in self.phases}

        if required_min > available:
            # Compress minimum greens to fit the available window.
            compressed = available / len(self.phases)
            return {p: round(max(0.0, compressed), 2) for p in self.phases}

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

        # If sum exceeds available (because mins pushed it), trim proportionally.
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

        # Final check: ensure no negative and round to 2 decimals while keeping the total close.
        greens = {p: round(max(0.0, g), 2) for p, g in greens.items()}
        rounded_sum = sum(greens.values())
        if abs(rounded_sum - available) >= 0.01 and rounded_sum > 0:
            delta = available - rounded_sum
            # adjust the phase with the largest green to absorb the rounding delta
            key = max(greens, key=greens.get)
            greens[key] = round(max(0.0, greens[key] + delta), 2)

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

    # ---------- Standardized dataset helpers ----------
    def estimate_per_phase_service_vph(
        self,
        cycle_length: float,
        greens: Dict[str, float],
    ) -> Dict[str, float]:
        """Estimate per-phase service rate (veh/hour) from cycle and green times.

        service_i ≈ saturation_flow × (green_i / cycle_length)
        """
        if cycle_length <= 0:
            return {p: 0.0 for p in self.phases}
        return {p: self.saturation_flow * (max(0.0, greens.get(p, 0.0)) / cycle_length) for p in self.phases}

    def build_roundabout_like_row(
        self,
        flows_vph: Dict[str, float],
        cycle_length: float,
        greens: Dict[str, float],
        window_s: float = 300.0,
        phase_order_for_arms: List[str] | None = None,
    ) -> Dict[str, float]:
        """Build a single-row dict shaped like the Roundabout per-window dataset.

        Returns keys matching parse_window_lines() output in streamlit_app:
        - time (mm:ss), arrivals, exits, throughput_vph, avg_delay_s, p95_delay_s,
          max_q_arm0..3

        Note: avg_delay_s, p95_delay_s, and max_q_* are not known from a static plan,
        so they are returned as None. This row is an approximation to let users export
        a "Roundabout-like" dataset for quick comparisons.
        """
        # Estimate total service capacity over the window
        per_phase_service_vph = self.estimate_per_phase_service_vph(cycle_length, greens)
        total_service_vph = sum(per_phase_service_vph.values())

        # Total arrivals and exits over the window
        total_arrivals = max(0.0, sum(float(flows_vph.get(p, 0.0)) for p in (phase_order_for_arms or self.phases)))
        arrivals_win = total_arrivals * (window_s / 3600.0)
        service_win = total_service_vph * (window_s / 3600.0)
        exits_win = min(arrivals_win, service_win)

        # Throughput in veh/hour for that window
        throughput_vph = (exits_win * 3600.0 / window_s) if window_s > 0 else 0.0

        # Map phases to arm indices (0..3) if provided, else use current phase order
        phase_order = phase_order_for_arms or self.phases
        # Fill max_q_* as None since queues are not simulated here
        row: Dict[str, float] = {
            "time": "00:00",
            "arrivals": int(round(arrivals_win)),
            "exits": int(round(exits_win)),
            "throughput_vph": int(round(throughput_vph)),
            "avg_delay_s": None,
            "p95_delay_s": None,
        }
        # Ensure 4 arms in order Arm0..Arm3 semantics
        for i in range(4):
            key = f"max_q_arm{i}"
            row[key] = None
        return row

    def build_roundabout_like_dataset(
        self,
        flows_vph: Dict[str, float],
        cycle_length: float,
        greens: Dict[str, float],
        horizon_s: float = 3600.0,
        window_s: float = 300.0,
        phase_order_for_arms: List[str] | None = None,
    ) -> List[Dict[str, float]]:
        """Build a list of per-window rows (Roundabout-like) covering horizon_s.

        Each window row contains aggregated arrivals and exits based on the
        input flows and the estimated per-phase service. This is a deterministic
        approximation (no stochastic arrivals) intended to produce a comparable
        CSV dataset for analysis and plotting.
        """
        phase_order = phase_order_for_arms or self.phases
        # Ensure consistent ordering for arms (if user passed a custom order)
        per_phase_service_vph = self.estimate_per_phase_service_vph(cycle_length, greens)

        # Per-window expected arrivals and service
        arrivals_per_window = {p: flows_vph.get(p, 0.0) * (window_s / 3600.0) for p in phase_order}
        service_per_window = {p: per_phase_service_vph.get(p, 0.0) * (window_s / 3600.0) for p in phase_order}

        rows: List[Dict[str, float]] = []
        n_windows = max(1, int(round(horizon_s / window_s)))
        queue = {p: 0.0 for p in phase_order}
        for w in range(n_windows):
            # accumulate arrivals into a running queue and serve what we can this window
            for p in phase_order:
                queue[p] += arrivals_per_window[p]
            exits_per_arm = {p: min(queue[p], service_per_window[p]) for p in phase_order}
            for p in phase_order:
                queue[p] = max(0.0, queue[p] - exits_per_arm[p])

            arrivals_win = sum(arrivals_per_window.values())
            exits_win = sum(exits_per_arm.values())

            throughput_vph = (exits_win * 3600.0 / window_s) if window_s > 0 else 0.0

            row: Dict[str, float] = {
                "time": f"{int((w*window_s)//60):02d}:{int((w*window_s)%60):02d}",
                "arrivals": int(round(arrivals_win)),
                "exits": int(round(exits_win)),
                "throughput_vph": int(round(throughput_vph)),
                "avg_delay_s": None,
                "p95_delay_s": None,
            }

            # per-arm max queue placeholders (use current queue as a proxy)
            for i in range(4):
                if i < len(phase_order):
                    phase = phase_order[i]
                    row[f"max_q_arm{i}"] = int(round(queue[phase]))
                else:
                    row[f"max_q_arm{i}"] = None

            rows.append(row)

        return rows

    def build_roundabout_like_dataset_stochastic(
        self,
        flows_vph: Dict[str, float],
        cycle_length: float,
        greens: Dict[str, float],
        horizon_s: float = 3600.0,
        window_s: float = 300.0,
        phase_order_for_arms: List[str] | None = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Build a list of per-window rows using stochastic (Poisson) arrivals and service.

        For each arm and window we sample arrivals ~ Poisson(lambda=flow_vph*(window_s/3600)).
        Service capacity per arm in a window is estimated from the signal plan and
        sampled as Poisson(service_rate*(window_s/3600)). Exits = min(arrivals, service).

        Returns a list of rows similar to `build_roundabout_like_dataset` but with
        per-window stochastic variability. Use `seed` for reproducible draws.
        """
        rng = random.Random(seed)

        phase_order = phase_order_for_arms or self.phases
        per_phase_service_vph = self.estimate_per_phase_service_vph(cycle_length, greens)

        # Helper: Poisson sampler using Knuth's algorithm (works fine for moderate lambda)
        def _poisson(lmbda: float) -> int:
            if lmbda <= 0:
                return 0
            import math as _math
            if lmbda > 50:
                # normal approximation for performance at large lambdas
                sample = rng.gauss(mu=lmbda, sigma=_math.sqrt(lmbda))
                return max(0, int(round(sample)))
            # Knuth's algorithm for moderate lambdas
            L = _math.exp(-lmbda)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= rng.random()
            return k - 1

        rows: List[Dict[str, float]] = []
        n_windows = max(1, int(round(horizon_s / window_s)))

        # Precompute per-window lambdas
        arrivals_lambda = {p: flows_vph.get(p, 0.0) * (window_s / 3600.0) for p in phase_order}
        service_lambda = {p: per_phase_service_vph.get(p, 0.0) * (window_s / 3600.0) for p in phase_order}
        queue = {p: 0.0 for p in phase_order}

        for w in range(n_windows):
            # sample arrivals and service per arm
            arrivals_per_arm = {p: _poisson(arrivals_lambda[p]) for p in phase_order}
            service_per_arm = {p: _poisson(service_lambda[p]) for p in phase_order}

            for p in phase_order:
                queue[p] += arrivals_per_arm[p]

            exits_per_arm = {p: min(queue[p], service_per_arm[p]) for p in phase_order}
            for p in phase_order:
                queue[p] = max(0.0, queue[p] - exits_per_arm[p])

            arrivals_win = sum(arrivals_per_arm.values())
            exits_win = sum(exits_per_arm.values())

            throughput_vph = (exits_win * 3600.0 / window_s) if window_s > 0 else 0.0

            row: Dict[str, float] = {
                "time": f"{int((w*window_s)//60):02d}:{int((w*window_s)%60):02d}",
                "arrivals": int(arrivals_win),
                "exits": int(exits_win),
                "throughput_vph": int(round(throughput_vph)),
                "avg_delay_s": None,
                "p95_delay_s": None,
            }

            # per-arm max queue placeholders (current queue as proxy) — could be extended to track queues
            for i in range(4):
                if i < len(phase_order):
                    phase = phase_order[i]
                    row[f"max_q_arm{i}"] = int(round(queue[phase]))
                else:
                    row[f"max_q_arm{i}"] = None

            rows.append(row)

        return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate signal plan and Roundabout-like dataset CSV")
    parser.add_argument("--horizon", type=float, default=3600.0, help="Total horizon in seconds (default 3600)")
    parser.add_argument("--window", type=float, default=300.0, help="Window size in seconds (default 300)")
    parser.add_argument("--stochastic", action="store_true", help="Enable stochastic sampling per window (Poisson)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for stochastic sampling")
    parser.add_argument("--out", type=str, default="signal_dataset.csv", help="Output CSV filename")
    parser.add_argument("--flows", type=str, default=None, help="Comma-separated flows as N=300,E=200,... (veh/hr)")
    parser.add_argument("--phases", type=str, default="N,E,S,W", help="Comma-separated phase names (default N,E,S,W)")
    parser.add_argument("--plot-plan", type=str, default=None, help="Optional PNG path to save green-time bar chart")
    parser.add_argument("--plot-dataset", type=str, default=None, help="Optional PNG path to plot arrivals/exits/throughput")
    args = parser.parse_args()

    # Parse flows
    if args.flows:
        flows = {}
        for pair in args.flows.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    flows[k.strip()] = float(v)
                except Exception:
                    flows[k.strip()] = 0.0
    else:
        flows = {"N": 300.0, "E": 200.0, "S": 250.0, "W": 150.0}

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    ctrl = ModifiedWebsterRoundabout(phases=phases)
    cycle, greens = ctrl.get_signal_plan(flows)

    print(f"Cycle: {cycle:.2f} s")
    print("Greens:")
    for p, g in greens.items():
        print(f"  {p}: {g:.2f} s")

    # Build dataset
    if args.stochastic:
        rows = ctrl.build_roundabout_like_dataset_stochastic(
            flows_vph=flows,
            cycle_length=cycle,
            greens=greens,
            horizon_s=float(args.horizon),
            window_s=float(args.window),
            phase_order_for_arms=phases,
            seed=args.seed,
        )
    else:
        rows = ctrl.build_roundabout_like_dataset(
            flows_vph=flows,
            cycle_length=cycle,
            greens=greens,
            horizon_s=float(args.horizon),
            window_s=float(args.window),
            phase_order_for_arms=phases,
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

    # Optional plotting helpers
    def _maybe_import_matplotlib():
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless backend for CLI use
            import matplotlib.pyplot as plt
            return plt
        except Exception as e:
            print(f"Skipping plot (matplotlib not available): {e}")
            return None

    if args.plot_plan:
        plt = _maybe_import_matplotlib()
        if plt:
            fig, ax = plt.subplots(figsize=(6, 4))
            phases_plot = list(greens.keys())
            values = [greens[p] for p in phases_plot]
            ax.bar(phases_plot, values, color="#4e79a7")
            ax.set_ylabel("Green time (s)")
            ax.set_xlabel("Phase")
            ax.set_title(f"Cycle={cycle:.1f}s (lost={float(len(phases_plot)*ctrl.lost_time_per_phase):.1f}s)")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            out_path = Path(args.plot_plan)
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved plan plot to {out_path}")

    if args.plot_dataset:
        plt = _maybe_import_matplotlib()
        if plt and not df.empty:
            time_index = df["time"]
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            axes[0].plot(time_index, df["arrivals"], label="arrivals", color="#59a14f")
            axes[0].plot(time_index, df["exits"], label="exits", color="#f28e2b")
            axes[0].plot(time_index, df["throughput_vph"], label="throughput_vph", color="#4e79a7")
            axes[0].set_ylabel("Vehicles / window")
            axes[0].legend(loc="best")
            axes[0].grid(alpha=0.3)

            q_cols = [c for c in df.columns if c.startswith("max_q_arm")]
            if q_cols:
                for col in q_cols:
                    axes[1].plot(time_index, df[col], label=col)
                axes[1].set_ylabel("Queue (veh)")
                axes[1].legend(loc="best", ncol=2)
                axes[1].grid(alpha=0.3)
            axes[1].set_xlabel("time (mm:ss)")
            fig.tight_layout()
            out_path = Path(args.plot_dataset)
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved dataset plot to {out_path}")
