import io
import re
import contextlib
from typing import List, Tuple

import streamlit as st
import pandas as pd

from traffic_signal import ModifiedWebsterRoundabout


def normalize_turning(p: Tuple[float, float, float]) -> Tuple[float, float, float]:
    a, b, c = p
    total = max(1e-9, a + b + c)
    return (a / total, b / total, c / total)


# First Streamlit command must be set_page_config
st.set_page_config(page_title="Roundabout Microsimulation", layout="wide")
st.title("Roundabout Traffic Microsimulation")
st.caption("Interactive wrapper around the Python simulator with 5‑minute summaries")
st.info("Status: app loaded. If you see a black screen, reload the page or check the Streamlit server logs in the terminal.")

# Simple, plain‑language help for non‑experts
with st.expander("Help (plain language)", expanded=False):
    st.markdown(
        """
        What does "Compute signal plan" do?
        - It takes the traffic you enter for each side and suggests a simple stoplight schedule.
        - You’ll get a total cycle time (how long one full round of greens takes) and
          how many seconds of green each side should get per cycle.

        Key terms in the results:
        - Cycle length: one full loop of the signal, in seconds.
        - Green time (green_s): how many seconds of green that side gets each cycle (usable green; lost time is separate).
        - Flow (flow_vph): the cars/hour you entered (λ × 3600).
        - Saturation flow: the max serving speed of a lane group if it stayed green the whole time (typical ≈ 1800 veh/h per lane).

        How it works (simple):
        1) Turn your arrivals (λ) into cars/hour for each road.
        2) Figure out how busy each road is vs. its serving speed; give busier roads more green.
        3) Pick a cycle that leaves room for the in‑between "lost time"; roundabouts use a slightly shorter cycle.
        4) Split the effective green (cycle − lost time) across roads, keeping a small minimum for each.

        How to use it:
        1) Enter the arrival rates (λ) for each arm. Tip: 0.20 ≈ 0.20 cars/second ≈ 720 cars/hour.
        2) Either turn on "Auto‑update signal plan" (recompute on any change) or click "Compute signal plan".
        3) Read the cycle time and the green seconds for N/E/S/W.
        4) Use the "Export (Roundabout‑like dataset)" table + download to get a CSV shaped like the simulator's per‑window output (delay/queues are placeholders).

        Notes:
        - This planner is advisory. It does not change the simulator yet—it just shows a suggested signal plan.
        - Arms map to directions as: Arm0→N, Arm1→E, Arm2→S, Arm3→W.
        """
    )


with st.sidebar:
    st.header("Simulation Controls")

    seed = st.number_input("seed", value=42, step=1)
    horizon = st.number_input("horizon (s)", value=3600.0, min_value=60.0, step=300.0)
    dt = st.number_input("dt (s)", value=0.2, min_value=0.05, step=0.05, format="%0.2f")
    report_every = st.number_input("report_every (s)", value=300.0, min_value=60.0, step=60.0)

    st.divider()
    st.subheader("Geometry (SimConfig.geo)")
    diameter = st.number_input("diameter (m)", value=45.0, min_value=5.0, step=1.0)
    lanes = st.selectbox("lanes", options=[1, 2], index=0)
    a_lat_max = st.number_input("a_lat_max (m/s²)", value=1.6, min_value=0.2, step=0.1)

    st.divider()
    st.subheader("Demand (SimConfig.demand)")
    st.caption("Poisson arrival rates λ (veh/s) for 4 arms")
    c1, c2 = st.columns(2)
    with c1:
        a0 = st.number_input("Arm 0 λ", value=0.18, min_value=0.0, step=0.01, format="%0.2f", help="Vehicles per second (e.g., 0.20 ≈ 720 veh/hr)")
        a1 = st.number_input("Arm 1 λ", value=0.12, min_value=0.0, step=0.01, format="%0.2f", help="Vehicles per second (e.g., 0.12 ≈ 432 veh/hr)")
    with c2:
        a2 = st.number_input("Arm 2 λ", value=0.20, min_value=0.0, step=0.01, format="%0.2f", help="Vehicles per second (e.g., 0.20 ≈ 720 veh/hr)")
        a3 = st.number_input("Arm 3 λ", value=0.15, min_value=0.0, step=0.01, format="%0.2f", help="Vehicles per second (e.g., 0.15 ≈ 540 veh/hr)")

    with st.expander("Arm legend", expanded=False):
        st.markdown(
            """
            - Arm 0 → North (N)
            - Arm 1 → East (E)
            - Arm 2 → South (S)
            - Arm 3 → West (W)
            """
        )

    st.caption("Turning probabilities turning_LTR=(L,T,R). Will be normalized to sum to 1.")
    L = st.number_input("L (left)", value=0.25, min_value=0.0, step=0.05)
    T = st.number_input("T (through)", value=0.55, min_value=0.0, step=0.05)
    R = st.number_input("R (right)", value=0.20, min_value=0.0, step=0.05)

    st.divider()
    st.subheader("Gap Acceptance (SimConfig.gaps)")
    crit_gap_mean = st.number_input("crit_gap_mean (s)", value=3.0, min_value=0.2, step=0.1)
    crit_gap_sd = st.number_input("crit_gap_sd (s)", value=0.6, min_value=0.0, step=0.1)
    followup_mean = st.number_input("followup_mean (s)", value=2.0, min_value=0.2, step=0.1)
    followup_sd = st.number_input("followup_sd (s)", value=0.3, min_value=0.0, step=0.1)

    st.divider()
    st.subheader("Driver/IDM (SimConfig.driver)")
    a_max = st.number_input("a_max (m/s²)", value=1.5, min_value=0.1, step=0.1)
    b_comf = st.number_input("b_comf (m/s²)", value=2.0, min_value=0.1, step=0.1)
    T_headway = st.number_input("T (desired headway, s)", value=1.2, min_value=0.1, step=0.1)
    tau = st.number_input("tau (s)", value=0.8, min_value=0.0, step=0.1)
    v0_ring = st.number_input("v0_ring (m/s)", value=12.0, min_value=1.0, step=0.5)

    st.divider()
    st.subheader("Traffic signal (Modified Webster)")
    sat_flow = st.number_input("Saturation flow (veh/hr)", value=1800.0, step=50.0)
    lost_time = st.number_input("Lost time per phase (s)", value=4.0, step=0.5)
    roundabout_factor = st.slider("Roundabout factor", 0.5, 1.0, 0.9, 0.05)

    auto_compute = st.checkbox("Auto-update signal plan", value=False)
    compute_signal = st.button("Compute signal plan") or auto_compute

    run_btn = st.button("Run simulation", type="primary")



def run_simulation() -> str:
    # Import simulator classes here to avoid heavy work during module import
    try:
        from Roundabout import (
            SimConfig,
            Geometry,
            DriverParams,
            GapParams,
            Demand,
            RoundaboutSim,
        )
    except Exception as e:
        st.error(f"Failed to import Roundabout simulator: {e}")
        return ""

    arr = [a0, a1, a2, a3]
    turn = normalize_turning((L, T, R))

    geo = Geometry(diameter=diameter, lanes=int(lanes), a_lat_max=a_lat_max)
    driver = DriverParams(a_max=a_max, b_comf=b_comf, T=T_headway, tau=tau, v0_ring=v0_ring)
    gaps = GapParams(
        crit_gap_mean=crit_gap_mean,
        crit_gap_sd=crit_gap_sd,
        followup_mean=followup_mean,
        followup_sd=followup_sd,
    )
    demand = Demand(arrivals=arr, turning_LTR=turn)
    cfg = SimConfig(
        seed=int(seed),
        horizon=float(horizon),
        dt=float(dt),
        report_every=float(report_every),
        geo=geo,
        driver=driver,
        gaps=gaps,
        demand=demand,
    )

    sim = RoundaboutSim(cfg)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.run()
    return buf.getvalue()


def parse_window_lines(output_text: str) -> pd.DataFrame:
    pattern = re.compile(
        r"^\[(?P<time>\d{2}:\d{2})\].*?arrivals=(?P<arr>\d+)\s+exits=(?P<ex>\d+)\s+throughput=(?P<thr>\d+)\s+veh/hr\s+avg_delay=(?P<avg>[0-9.]+)s\s+p95=(?P<p95>[0-9.]+)s\s+max_q=\[(?P<q>[^\]]*)\]",
        re.M,
    )
    rows: List[dict] = []
    for m in pattern.finditer(output_text):
        q = [int(x.strip()) for x in m.group("q").split(",") if x.strip()]
        rows.append(
            {
                "time": m.group("time"),
                "arrivals": int(m.group("arr")),
                "exits": int(m.group("ex")),
                "throughput_vph": int(m.group("thr")),
                "avg_delay_s": float(m.group("avg")),
                "p95_delay_s": float(m.group("p95")),
                "max_q_arm0": q[0] if len(q) > 0 else 0,
                "max_q_arm1": q[1] if len(q) > 1 else 0,
                "max_q_arm2": q[2] if len(q) > 2 else 0,
                "max_q_arm3": q[3] if len(q) > 3 else 0,
            }
        )
    return pd.DataFrame(rows)


if run_btn:
    with st.spinner("Running simulation..."):
        out_text = run_simulation()

    st.subheader("Text output")
    st.code(out_text, language="text")

    st.download_button(
        "Download output.txt", data=out_text, file_name="simulation_output.txt"
    )

    df = parse_window_lines(out_text)
    if not df.empty:
        st.subheader("Per-window metrics")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(df.set_index("time")["throughput_vph"], height=240)
            st.caption("Throughput (veh/hr) per reporting window")
        with c2:
            st.line_chart(df.set_index("time")["avg_delay_s"], height=240)
            st.caption("Average delay (s) per reporting window")

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No per-window lines parsed; try a longer horizon or smaller report interval."
        )


# If the user requested a signal plan, compute and show it in the main area
if compute_signal:
    # Use cardinal labels for readability in results
    arm_labels = ["N", "E", "S", "W"]
    flows_vph = {
        "N": a0 * 3600.0,
        "E": a1 * 3600.0,
        "S": a2 * 3600.0,
        "W": a3 * 3600.0,
    }
    phases = arm_labels
    ctrl = ModifiedWebsterRoundabout(
        phases=phases,
        saturation_flow=sat_flow,
        lost_time_per_phase=lost_time,
        roundabout_factor=roundabout_factor,
    )
    cycle, greens = ctrl.get_signal_plan(flows_vph)

    st.subheader("Signal plan (Modified Webster)")
    st.metric("Cycle length (s)", f"{cycle:.1f}")

    g_df = pd.DataFrame.from_dict(
        greens, orient="index", columns=["green_s"]
    )  # green_s = green seconds per cycle
    g_df["flow_vph"] = pd.Series(
        flows_vph
    )  # flow_vph = planning flow (cars/hour) used to size greens
    st.table(g_df)
    st.bar_chart(g_df["green_s"])

    # Standardized dataset export to match Roundabout per-window schema (approximate)
    st.markdown("**Export (Roundabout-like dataset)**")
    window_s = report_every  # use the same window size as the simulator summary

    dataset_mode = st.selectbox(
        "Dataset mode",
        [
            "Single window (summary)",
            "Deterministic multi-window",
            "Stochastic multi-window",
        ],
        index=0,
    )

    if dataset_mode == "Single window (summary)":
        rb_row = ctrl.build_roundabout_like_row(
            flows_vph=flows_vph,
            cycle_length=cycle,
            greens=greens,
            window_s=float(window_s),
            phase_order_for_arms=phases,
        )
        rb_df = pd.DataFrame([rb_row])
        st.dataframe(rb_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download signal_plan_window.csv",
            data=rb_df.to_csv(index=False),
            file_name="signal_plan_window.csv",
            mime="text/csv",
        )

    else:
        horizon_s = float(horizon)
        window_s = float(window_s)
        if dataset_mode == "Deterministic multi-window":
            rows = ctrl.build_roundabout_like_dataset(
                flows_vph=flows_vph,
                cycle_length=cycle,
                greens=greens,
                horizon_s=horizon_s,
                window_s=window_s,
                phase_order_for_arms=phases,
            )
        else:
            seed_val = st.number_input("Random seed (int)", value=42, step=1)
            rows = ctrl.build_roundabout_like_dataset_stochastic(
                flows_vph=flows_vph,
                cycle_length=cycle,
                greens=greens,
                horizon_s=horizon_s,
                window_s=window_s,
                phase_order_for_arms=phases,
                seed=int(seed_val),
            )

        df_rows = pd.DataFrame(rows)
        st.dataframe(df_rows, use_container_width=True)
        csv_bytes = df_rows.to_csv(index=False)
        file_name = (
            "signal_stochastic.csv" if dataset_mode == "Stochastic multi-window" else "signal_deterministic.csv"
        )
        st.download_button("Download dataset CSV", data=csv_bytes, file_name=file_name, mime="text/csv")

        # Quick visuals for the exported dataset
        if not df_rows.empty:
            with st.expander("Charts", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Arrivals / Exits / Throughput")
                    st.line_chart(
                        df_rows.set_index("time")[["arrivals", "exits", "throughput_vph"]],
                        height=280,
                    )
                with c2:
                    st.subheader("Queue proxy per arm")
                    queue_cols = [c for c in df_rows.columns if c.startswith("max_q_arm")]
                    if queue_cols:
                        st.line_chart(df_rows.set_index("time")[queue_cols], height=280)
                    else:
                        st.info("Queue columns not present in dataset.")

st.markdown(
    """
    Tips:
    - "Report every" sets the 5‑minute window size (default 300 s).
    - Turning inputs are normalized to sum to 1.
    - If arrivals are very low, some windows may be empty (zero throughput/delay).
    """
)

