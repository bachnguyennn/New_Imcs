import io
import re
import contextlib
from typing import List, Tuple

import streamlit as st
import pandas as pd

from Roundabout import (
    SimConfig,
    Geometry,
    DriverParams,
    GapParams,
    Demand,
    RoundaboutSim,
)


def normalize_turning(p: Tuple[float, float, float]) -> Tuple[float, float, float]:
    a, b, c = p
    total = max(1e-9, a + b + c)
    return (a / total, b / total, c / total)


st.set_page_config(page_title="Roundabout Microsimulation", layout="wide")
st.title("Roundabout Traffic Microsimulation")
st.caption("Interactive wrapper around the Python simulator with 5‑minute summaries")

with st.sidebar:
    st.header("Simulation Controls")

    seed = st.number_input("Seed", value=42, step=1)
    horizon = st.number_input("Horizon (s)", value=3600.0, min_value=60.0, step=300.0)
    dt = st.number_input("Time step dt (s)", value=0.2, min_value=0.05, step=0.05, format="%0.2f")
    report_every = st.number_input("Report every (s)", value=300.0, min_value=60.0, step=60.0)

    st.divider()
    st.subheader("Geometry")
    diameter = st.number_input("Diameter (m)", value=45.0, min_value=5.0, step=1.0)
    lanes = st.selectbox("Lanes", options=[1, 2], index=0)
    a_lat_max = st.number_input("Lateral accel cap (m/s²)", value=1.6, min_value=0.2, step=0.1)

    st.divider()
    st.subheader("Demand")
    st.caption("Poisson arrival rates λ (veh/s) for 4 arms")
    c1, c2 = st.columns(2)
    with c1:
        a0 = st.number_input("Arm 0 λ", value=0.18, min_value=0.0, step=0.01, format="%0.2f")
        a1 = st.number_input("Arm 1 λ", value=0.12, min_value=0.0, step=0.01, format="%0.2f")
    with c2:
        a2 = st.number_input("Arm 2 λ", value=0.20, min_value=0.0, step=0.01, format="%0.2f")
        a3 = st.number_input("Arm 3 λ", value=0.15, min_value=0.0, step=0.01, format="%0.2f")

    st.caption("Turning probabilities (L/T/R). Will be normalized to sum to 1.")
    L = st.number_input("Left (L)", value=0.25, min_value=0.0, step=0.05)
    T = st.number_input("Through (T)", value=0.55, min_value=0.0, step=0.05)
    R = st.number_input("Right (R)", value=0.20, min_value=0.0, step=0.05)

    st.divider()
    st.subheader("Gap Acceptance")
    crit_gap_mean = st.number_input("Critical gap mean (s)", value=3.0, min_value=0.2, step=0.1)
    crit_gap_sd = st.number_input("Critical gap SD (s)", value=0.6, min_value=0.0, step=0.1)
    followup_mean = st.number_input("Follow-up mean (s)", value=2.0, min_value=0.2, step=0.1)
    followup_sd = st.number_input("Follow-up SD (s)", value=0.3, min_value=0.0, step=0.1)

    st.divider()
    st.subheader("Driver/IDM")
    a_max = st.number_input("a_max (m/s²)", value=1.5, min_value=0.1, step=0.1)
    b_comf = st.number_input("b_comf (m/s²)", value=2.0, min_value=0.1, step=0.1)
    T_headway = st.number_input("Desired headway T (s)", value=1.2, min_value=0.1, step=0.1)
    tau = st.number_input("Reaction delay τ (s)", value=0.8, min_value=0.0, step=0.1)
    v0_ring = st.number_input("Desired free-flow v0 (m/s)", value=12.0, min_value=1.0, step=0.5)

    run_btn = st.button("Run simulation", type="primary")


def run_simulation() -> str:
    arr = [a0, a1, a2, a3]
    turn = normalize_turning((L, T, R))

    geo = Geometry(diameter=diameter, lanes=int(lanes), a_lat_max=a_lat_max)
    driver = DriverParams(a_max=a_max, b_comf=b_comf, T=T_headway, tau=tau, v0_ring=v0_ring)
    gaps = GapParams(crit_gap_mean=crit_gap_mean, crit_gap_sd=crit_gap_sd,
                     followup_mean=followup_mean, followup_sd=followup_sd)
    demand = Demand(arrivals=arr, turning_LTR=turn)
    cfg = SimConfig(seed=int(seed), horizon=float(horizon), dt=float(dt), report_every=float(report_every),
                    geo=geo, driver=driver, gaps=gaps, demand=demand)

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
        rows.append({
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
        })
    return pd.DataFrame(rows)


if run_btn:
    with st.spinner("Running simulation..."):
        out_text = run_simulation()

    st.subheader("Text output")
    st.code(out_text, language="text")

    st.download_button("Download output.txt", data=out_text, file_name="simulation_output.txt")

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
        st.info("No per-window lines parsed; try a longer horizon or smaller report interval.")


st.markdown(
    """
    Tips:
    - "Report every" sets the 5‑minute window size (default 300 s).
    - Turning inputs are normalized to sum to 1.
    - If arrivals are very low, some windows may be empty (zero throughput/delay).
    """
)


