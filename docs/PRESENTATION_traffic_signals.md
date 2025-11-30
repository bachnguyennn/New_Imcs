---
title: Traffic Signals — 5-minute Slides
author: Project: Roundabout / Modified Webster Controller
date: 2025-11-26
---

## Slide 1 — Title

Traffic Signals: Modified Webster Controller

Speaker notes:
- Quick intro: this deck focuses only on the traffic-signal work (controller, dataset export, Streamlit demo).

---

## Slide 2 — Why traffic signals?

- Traffic signals regulate conflicting flows, improve safety and throughput.
- For this project we need a simple, reproducible signal plan generator that matches the Roundabout dataset format.

Speaker notes:
- Emphasize repeatability (deterministic vs stochastic exports) so downstream models can compare apples-to-apples.

---

## Slide 3 — Problem statement

- Given per-arm vehicle flows (vph) and a few driver/gap parameters, compute:
  - cycle length (s)
  - per-arm green times (s)
  - produce CSV rows in the "Roundabout-like" format (one row or many windows)

Speaker notes:
- Mention constraints: lost time per phase, min green per arm. Output must be compatible with the existing Roundabout dataset schema.

---

## Slide 4 — Algorithm: Modified Webster

- Based on Webster's formula (cycle length ~ function of critical flow ratios) with a reduction for roundabout-equivalent behavior.
- Steps:
  1. Compute critical flow ratios for each phase.
  2. Compute ideal cycle length (Webster) with safety factor.
  3. Allocate green times proportionally, enforce min-green, round to 0.1s.

Speaker notes:
- This is intentionally simple and deterministic; good trade-off between realism and reproducibility.

---

## Slide 5 — Implementation notes (files & API)

- `traffic_signal.py`:
  - ModifiedWebsterRoundabout class
  - Methods: compute_cycle_length(), allocate_green_times(), get_signal_plan(), build_roundabout_like_dataset(), build_roundabout_like_dataset_stochastic()
  - CLI for CSV export (deterministic / stochastic)
- `streamlit_app.py`:
  - UI to enter flows and driver/gap params, compute plan, preview dataset, download CSV.

Speaker notes:
- The code is seedable for stochastic exports; Poisson arrivals used for the stochastic mode.

---

## Slide 6 — Dataset modes

- Single-row: one aggregated row that matches the Roundabout single-window summary.
- Deterministic multi-window: repeated identical-window rows for the horizon/window size.
- Stochastic multi-window: Poisson-sampled per-window arrivals per arm, seedable for reproducibility.

Speaker notes:
- Use deterministic for baseline experiments; use stochastic to test variability and robustness.

---

## Slide 7 — How to demo (Streamlit)

1. From the project root run:

```
streamlit run streamlit_app.py
```

2. In the app sidebar: enter flows (vph) for each arm, set gaps/lost time, and press Compute.
3. Use the Dataset export panel to choose Single / Deterministic / Stochastic and download the CSV.

Speaker notes:
- Streamlit performs a lazy import of the heavy simulator so the UI starts quickly. Make sure to run the Streamlit entry `streamlit_app.py` (not `traffic_signal.py`).

---

## Slide 8 — Sample results (quick)

- Example input: N=800, E=450, S=650, W=300 (vph)
- Example output: Cycle ~ 52 s, Greens: N 12.1 s, E 8.0 s, S 10.1 s, W 6.0 s

Speaker notes:
- These numbers are illustrative — exact values depend on lost times and min-green settings. CLI tests write CSVs to /tmp when used from the terminal.

---

## Slide 9 — Limitations & next steps

- Current model is a capacity-based approximation (no per-vehicle microsimulation).
- Next steps:
  - Integrate green plan into the Roundabout microsimulator to compute per-window delay/queue.
  - Replace the small Poisson sampler with numpy vectorized draws for large horizons/lambdas.
  - Add per-arm arrival/departure logging to exported CSVs if desired.

Speaker notes:
- These are low-risk improvements; I can implement any of them on request.

---

## Slide 10 — Takeaways & Q&A

- We have a compact, deterministic Modified-Webster controller that:
  - Produces Roundabout-compatible CSV datasets (single, deterministic multi-window, stochastic multi-window)
  - Is seedable for reproducible experiments
  - Is integrated into Streamlit for quick demo and downloads

Speaker notes:
- Invite questions and propose a short live demo if time permits.
