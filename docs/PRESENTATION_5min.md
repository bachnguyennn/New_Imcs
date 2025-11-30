# 5-minute Presentation — Traffic Signal Integration (Modified Webster)

Slide 1 — Title (30s)
- Title: "Roundabout Signal Planning: Method & Quick Results"
- Subtitle: Modified Webster controller + Streamlit demo
- Presenter: (your name) — Date: 2025-11-26

Speaker notes (30s):
- One-sentence goal: explain how we produced advisory fixed-time signal plans for a 4-arm roundabout and provide a downloadable, Roundabout-like dataset for analysis.

---

Slide 2 — Problem & Objective (40s)
- Problem: No quick, repeatable way to produce fixed-time signal plans and dataset exports for our roundabout simulator.
- Objective: Implement a lightweight Modified-Webster planner, integrate it into the Streamlit UI, and enable deterministic or stochastic per-window dataset exports (CSV).

Speaker notes (40s):
- Emphasize user pain: analysts wanted a quick way to compare signal plans vs. simulator outputs without running long microsims.

---

Slide 3 — Methodology (90s)
- Controller: Modified Webster adapted for roundabouts
  - Compute critical flow ratios y_i = flow_i / saturation_flow
  - Webster baseline: C0 = (1.5·L + 5) / (1 − Y), with L = total lost time
  - Apply roundabout_factor (e.g., 0.9) to shorten cycle for compact geometry
  - Allocate greens: g_i = (y_i / Y)·(C − L) with min-green enforcement and re-normalization
- Export pipeline:
  - Single-row summary: per-window totals (arrivals, exits, throughput)
  - Deterministic multi-window: arrivals = flow_vph*(window_s/3600), exits = min(arrivals, estimated service)
  - Stochastic multi-window: Poisson-sampled arrivals and sampled service (seedable)

Speaker notes (90s):
- Walk through the small algorithmic steps and the intuition behind Webster and why we reduce the cycle for roundabouts.
- Explain deterministic vs. stochastic exports and why stochastic sampling is useful for quick uncertainty checks.

---

Slide 4 — Implementation & UI (40s)
- Code: `traffic_signal.py` (ModifiedWebsterRoundabout) — now includes dataset generators:
  - `build_roundabout_like_row`
  - `build_roundabout_like_dataset` (deterministic)
  - `build_roundabout_like_dataset_stochastic` (Poisson sampling, seedable)
- UI: `streamlit_app.py` — Compute button; Dataset mode selector; CSV download

Speaker notes (40s):
- Show a quick screenshot (or demo) of the Streamlit controls: λ inputs, sat_flow, lost_time, Compute, Dataset mode, Download.
- Mention lazy import of the simulator to avoid blocking the UI.

---

Slide 5 — Key Results (40s)
- Example (default inputs):
  - Cycle ≈ 52.2 s; Greens: N=12.1s, E=8.0s, S=10.1s, W=6.0s
  - Multi-window CSV: 1-hour horizon with 5-min windows → 12 rows (deterministic or stochastic)
- Why useful:
  - Analysts can quickly download test datasets for downstream plotting/ML.
  - Stochastic sampling gives realistic variability for sensitivity checks without running the full microsim.

Speaker notes (40s):
- Read the numeric example and explain that CSVs are ready for side-by-side comparison with microsim outputs.

---

Slide 6 — Limitations & Next Steps (40s)
- Limitations:
  - Delay and max-queue fields are placeholders; accurate estimates require microsimulation or queue tracking.
  - Knuth Poisson sampler is fine for small-to-moderate lambdas; for large/high-throughput scenarios, numpy is faster.
- Next steps (suggested):
  - Wire the plan into the Roundabout simulator to export real per-window delay/queue metrics.
  - Offer a Streamlit toggle to run a short microsim with the plan for true performance metrics.
  - Optionally switch to numpy for faster stochastic sampling and add per-arm arrivals/exits columns.

Speaker notes (40s):
- End with one slide of recommended immediate actions and invite questions.

---

Appendix: Short demo script (optional handout)
```bash
# Deterministic CSV (1 hour, 5-min windows)
python traffic_signal.py --horizon 3600 --window 300 --out signal_det.csv

# Stochastic CSV (seeded)
python traffic_signal.py --horizon 3600 --window 300 --stochastic --seed 123 --out signal_stoch.csv
```

Speaker note: Hand this out or paste into chat for quick replication.
