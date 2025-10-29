"""
Roundabout Traffic Microsimulation — One‑Hour, 5‑Minute Summaries
=================================================================

Design goals for this redo
--------------------------
- Exactly **1 hour** of simulation by default.
- **Only 12 summary lines**: one at the end of each 5‑minute window.
- Lines report **per‑window** stats (not cumulative averages):
  arrivals, exits, window throughput (veh/hr), window avg delay, p95 delay,
  and per‑arm window max queue lengths.
- Final hourly summary at the end.

Behavioral model
----------------
- **Arrivals**: Poisson per arm (veh/s); turning ratios L/T/R.
- **Entry**: Gap acceptance using per‑vehicle *critical gap* and *follow‑up* headway.
- **In‑ring**: IDM with reaction delay τ (DDE style via a short state history).
- **Geometry**: Diameter caps ring speed by lateral‑acceleration limit.

Example
-------
    python roundabout_sim.py \
      --seed 42 --horizon 3600 --report-every 300 \
      --diameter 45 --lanes 1 \
      --arrival 0.18 0.12 0.20 0.15 \
      --turning 0.25 0.55 0.20 \
      --crit-gap-mean 3.0 --crit-gap-sd 0.6 \
      --followup-mean 2.0 --followup-sd 0.3
"""

from __future__ import annotations  # Enable postponed evaluation of type hints (forward references)
from dataclasses import dataclass, field  # For concise data containers with default behavior
from typing import List, Deque, Dict, Optional, Tuple  # Type annotations for clarity and tooling
import math  # Math utilities (pi, sqrt, log, etc.)
import random  # RNG for arrivals and behavioral draws
import argparse  # CLI argument parsing for configuration
import statistics as stats  # For quantiles and averages on delays
from collections import deque  # Efficient FIFO queues for per-arm waiting lines and history

# ------------------------------ helpers ------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x # Clamp x into [lo, hi] (a range).
#clamp is here to keep physically impossible values from occurring due to numerical integration and
#model noise.

# distance ahead on ring

def ahead_distance(a: float, b: float, C: float) -> float:
    d = b - a #signed arc distance from point a to b along the ring.
    if d < 0: 
        d += C # Wrap-around if negative by adding circumference
    return d # Non-negative distance ahead
'''
Forward gap from entry to the next car ahead: ahead_distance(entry, car, C)

Backward gap from entry to the nearest car behind: ahead_distance(car, entry, C)
This gives two directed gaps.
'''

# pretty time

def mmss(t: float) -> str:
    m, s = divmod(int(round(t)), 60) # Convert seconds to integer minutes and seconds
    return f"{m:02d}:{s:02d}" # Zero-padded mm:ss string

# ------------------------------ config -------------------------------------

# Intelligent Driver Model (IDM) and driver behavior.
"""
As a car-following model, the IDM describes the dynamics of the positions and velocities of 
single vehicles.

The influencing factors of the IDM are the speed of the vehicle, 
the bumper-to-bumper gap to the leading vehicle, and the relative speed of the two vehicles. 
The model output is the acceleration chosen by the driver for that situation. 
The model parameters describe the driving style.
See: https://en.wikipedia.org/wiki/Intelligent_driver_model
"""
@dataclass
class DriverParams:
    a_max: float = 1.5 #comfortable acceleration in m/s^2
    b_comf: float = 2.0 #comfortable braking in m/s^2
    delta: int = 4 # Acceleration exponent in IDM (typical value ≈4)
    s0: float = 2.0 # Minimum desired gap (m) at standstill
    T: float = 1.2 # Desired time headway (s) in the ring
    tau: float = 0.8 # Reaction delay (s) used for DDE-like behavior
    v0_ring: float = 12.0 # Desired/target free-flow speed on ring (m/s)  ~43.2 km/h

@dataclass 
class GapParams: #gap-acceptance behavior
    crit_gap_mean: float = 3.0 # Mean critical gap for first entry (s)
    crit_gap_sd: float = 0.6 # Std dev of critical gap (s)
    '''
    Critical gap (first car in a platoon): 
    the minimum time gap in the circulating flow that a driver 
    requires to accept a merge from the stop line. Think: 
    “how big a hole in traffic do I need to go first?”
    '''
    followup_mean: float = 2.0 # Mean follow-up headway for subsequent vehicles (s)
    followup_sd: float = 0.3 # Std dev of follow-up headway (s)
    '''
    Follow-up headway (subsequent cars in the same platoon): 
    the time gap between successive entries once the first car has broken into the stream. 
    Think: “how quickly can the next car tuck in behind the first?”
    '''
@dataclass
class Geometry:
    diameter: float = 45.0 # Inscribed circle diameter (m)
    lanes: int = 1 # Number of circulating lanes (1 or 2)
    a_lat_max: float = 1.6 # Lateral acceleration comfort/limit (m/s^2) for curve speed cap
    #ring speed is capped by vmax ~= sqrt(a_lat_max * R) where R = diameter/2
    #which is derived from a_lat = v^2 / R 

    def circumference(self) -> float: # Ring circumference C = π·D
        return math.pi * self.diameter

    def ring_vmax(self) -> float:
        R = max(0.1, self.diameter / 2.0) # Use radius; guard against tiny values
        return math.sqrt(max(0.1, self.a_lat_max) * R) # v_max ≈ sqrt(a_lat_max · R)

@dataclass
class Demand:
    arrivals: List[float] = field(default_factory=lambda: [0.18, 0.12, 0.20, 0.15]) # Poisson rates per arm (veh/s)
    turning_LTR: Tuple[float, float, float] = (0.25, 0.55, 0.20) # Probabilities for Left/Through/Right

@dataclass
class SimConfig:
    seed: int = 42 # RNG seed for reproducibility
    horizon: float = 3600.0 # Total simulation time horizon in seconds (default 1 hour)
    dt: float = 0.2 # Simulation time step in seconds
    report_every: float = 300.0 # Reporting interval in seconds (default 5 minutes)
    geo: Geometry = field(default_factory=Geometry) # Geometry parameters
    driver: DriverParams = field(default_factory=DriverParams)  # Driver/IDM parameters
    gaps: GapParams = field(default_factory=GapParams) # Gap acceptance parameters
    demand: Demand = field(default_factory=Demand) # Demand parameters (arrival + turning ratios) 

# ------------------------------ entities -----------------------------------

_vehicle_id_ctr = 0 # Global counter to assign unique vehicle IDs

def _next_vid() -> int:
    global _vehicle_id_ctr # Use module-level counter
    _vehicle_id_ctr += 1 # Increment on each new vehicle creation
    return _vehicle_id_ctr # Return unique integer ID

@dataclass
class Vehicle:
    id: int # Unique vehicle ID
    origin: int # Entry arm index (0 to 3)
    target_exit: int # Exit arm index (0 to 3)
    '''
    Right = next exit = +1 step

    Through = opposite exit = +2 steps

    Left = third exit ahead = +3 steps
    '''
    t_create: float # Timestamp when vehicle object was created
    t_queue_start: float # Timestamp when vehicle joined queue
    crit_gap: float # Drawn critical gap for first entry (s) (drawn = randomly sampled)
    followup: float # Drawn follow-up headway (s) (followup = max(0.2, random.gauss(followup_mean, followup_sd)))
    tau: float # Driver's reaction delay used for DDE snapshot (s)
    lane_choice: int = 0 # Selected circulating lane index (0 or 1) 
    #for 2 lanes 0 = outer lane 1 = inner lane

    in_ring: bool = False # Flag indicating whether the vehicle has merged into ring
    pos: float = 0.0 # Position along ring (m), modulo circumference
    speed: float = 0.0 # Current speed (m/s)
    t_enter_ring: Optional[float] = None # Timestamp when vehicle entered ring

# ------------------------------ simulator ----------------------------------

class RoundaboutSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg # Store simulation configuration
        random.seed(cfg.seed) # Seed RNG for reproducible stochasticity

        # geometry
        self.C = cfg.geo.circumference() # Precompute circumference of ring
        self.N = 4 # Number of arms (fixed at 4)
        self.entry_pos = [i * self.C / self.N for i in range(self.N)] # Equally spaced entry positions
        self.exit_pos = self.entry_pos[:] # For single-lane, exits align with entries
        '''
        On this symmetric, single-lane ring we place 4 entry points equally spaced. 
        The exit for each arm is at the same angular position as that arm’s entry 
        (just the other side of the splitter island in reality, but same angle on the circle model). 
        So we reuse those positions.
        '''
        self.vmax_ring = min(cfg.driver.v0_ring, cfg.geo.ring_vmax()) # Speed cap: desired vs curve limit

        # state
        self.queues: List[Deque[Vehicle]] = [deque() for _ in range(self.N)] # Waiting queues per arm
        self.next_arrival: List[float] = [self._next_arrival_time(i, 0.0) for i in range(self.N)] # Next arrival times
        self.next_needed_headway: List[float] = [0.0] * self.N # Per-arm needed headway (crit gap or follow-up)
        self.lanes: List[List[Vehicle]] = [[] for _ in range(cfg.geo.lanes)] # Circulating lane vehicle lists
        '''
        When a car merges, it’s appended to self.lanes[lane_idx].

        On each step, the sim sorts each lane by position, updates motion (IDM),
        and checks exits using that lane’s list.

        Gap checks (_time_to_nearest_ring_vehicle, _space_ok_at_merge) 
        look into the specific lane list to find cars ahead/behind.
        '''

        # DDE history buffer: id -> (pos, speed)
        self.history: Deque[Dict[int, Tuple[float, float]]] = deque() # Past states for reaction delay snapshots
        '''
        self.history is a rolling timeline (a deque) of snapshots.

        Each snapshot is a dict mapping vehicle_id -> (position, speed) at a past step.
        '''
        self.max_hist_len = int(math.ceil(max(0.1, cfg.driver.tau) / cfg.dt)) + 2 # Buffer length to cover tau
        '''
        How many snapshots to keep.

        Compute the number of steps needed to cover the driver’s reaction delay tau, 
        given the time step dt.

        Example: tau=0.8 s, dt=0.2 s → tau/dt = 4 steps → keep ~4 + 2 = 6 snapshots.
        '''

        # time
        self.t = 0.0 # Simulation clock in seconds

        # cumulative metrics
        self.total_arrivals = 0 # Count of all arrivals generated
        self.total_exits = 0 # Count of all vehicles that have exited
        self.delays_all: List[float] = []  # Entry delays for all vehicles over simulation
        self.max_queue_len_global = [0] * self.N  # Max observed queue per arm over entire run

        # window metrics (reset each 5‑min window)
        self.win_arrivals = 0 # Arrivals within current reporting window
        self.win_exits = 0 # Exits within current reporting window
        self.win_delays: List[float] = [] # Entry delays within current window
        self.win_queue_max = [0] * self.N # Max queue length per arm within current window

    # --------------------------- random draws --------------------------------
    def _next_arrival_time(self, arm: int, now: float) -> float:
        lam = max(1e-9, self.cfg.demand.arrivals[arm]) # Poisson rate λ for this arm; guard tiny
        return now + random.expovariate(lam) # Next arrival = now + Exp(λ)
    '''
    random.expovariate(lam) needs lam > 0.

    If the user passes 0 or a tiny value, we replace it with 1e-9 so the draw won’t 
    crash or produce absurd waits.
    '''

    def _draw_turn_steps(self) -> int:
        L, T, R = self.cfg.demand.turning_LTR # Probabilities for Left/Through/Right
        u = random.random() # Uniform draw in [0,1)
        if u < R: 
            return 1  # Right = next exit
        elif u < R + T:
            return 2  # Through = opposite
        else:
            return 3  # Left = third exit

    def _draw_crit_gap(self) -> float:
        m, s = self.cfg.gaps.crit_gap_mean, self.cfg.gaps.crit_gap_sd # Mean and SD of critical gap
        mu = math.log((m*m) / math.sqrt(s*s + m*m))  # Lognormal μ from mean/SD
        sigma = math.sqrt(math.log(1 + (s*s) / (m*m))) # Lognormal σ from mean/SD
        return max(0.2, random.lognormvariate(mu, sigma)) # Draw, with lower bound to avoid degenerate values

    def _draw_followup(self) -> float:
        m, s = self.cfg.gaps.followup_mean, self.cfg.gaps.followup_sd # Mean and SD for follow-up
        return max(0.2, random.gauss(m, s)) # Gaussian draw, bounded below. Gaussian = normal distribution
    '''
    Follow-up headways (time between cars in the same platoon) are usually clustered around a typical 
    value (e.g., ~2 s) with roughly symmetric variation: a bit shorter or longer than average.
    '''
    def _choose_lane(self, turn_steps: int) -> int:
        if self.cfg.geo.lanes == 1:
            return 0 # Only one circulating lane available
        return 0 if turn_steps <= 2 else 1 # Simple rule: left turns prefer inner lane in 2-lane case
    

    # ----------------------------- history -----------------------------------
    def _push_history(self) -> None:
        snap: Dict[int, Tuple[float, float]] = {} # Build snapshot mapping id -> (pos, speed)
        for lane in self.lanes:
            for v in lane:
                snap[v.id] = (v.pos, v.speed) # Record state for each vehicle currently in ring
        self.history.append(snap) # Append snapshot to history buffer
        if len(self.history) > self.max_hist_len:
            self.history.popleft() # Keep buffer bounded to the needed length
            #In other words, don't let the history grow indefinitely; remove oldest snapshot 
            # if over limit.

    def _snapshot(self, steps_back: int) -> Dict[int, Tuple[float, float]]:
        if steps_back <= 0 or steps_back > len(self.history):
            return {}  # If delay not available, return empty snapshot
        return list(self.history)[-steps_back] # Retrieve snapshot approximately tau seconds ago

    # ------------------------------ mechanics --------------------------------
    def _spawn_arrivals(self) -> None:
        for i in range(self.N):  # Iterate over each arm
            if self.t >= self.next_arrival[i]: # If it's time for the next arrival on arm i
                ts = self._draw_turn_steps() # Randomly choose turn as steps to exit
                target = (i + ts) % self.N # Compute target exit arm index around ring
                v = Vehicle(
                    id=_next_vid(), origin=i, target_exit=target, # Unique ID and origin/exit
                    t_create=self.t, t_queue_start=self.t, # Timestamps for creation and queuing
                    crit_gap=self._draw_crit_gap(), followup=self._draw_followup(), # Behavior draws
                    tau=self.cfg.driver.tau, lane_choice=self._choose_lane(ts) # Reaction and lane selection
                )
                self.queues[i].append(v) # Place new vehicle at end of arm i queue
                self.total_arrivals += 1 # Update global arrival count
                self.win_arrivals += 1 # Update window arrival count
                self.next_arrival[i] = self._next_arrival_time(i, self.t) # Schedule next arrival time

    def _time_to_nearest_ring_vehicle(self, entry_pos: float, lane_index: int) -> float:
        lane = self.lanes[lane_index] # Select lane to check for gaps
        if not lane:
            return float('inf') # If no vehicles in lane, infinite time to next arrival -> immediate merge possible
        best = float('inf')  # Initialize minimum time until a vehicle reaches entry
        #“How soon will each of them (in the ring) reach this entry point?”
        for v in lane:
            d = ahead_distance(v.pos, entry_pos, self.C) # Distance from vehicle v to entry point ahead
            v_eff = max(0.1, v.speed) # Avoid divide-by-zero by imposing min speed
            best = min(best, d / v_eff) # Track smallest time-to-arrival at entry
        return best # Time until the closest ring vehicle reaches entry

    def _space_ok_at_merge(self, lane_index: int, entry_pos: float, min_gap: float) -> bool:
        lane = self.lanes[lane_index] # Lane to evaluate for space to merge
        if not lane:
            return True # If lane empty, space is sufficient
        d_ahead = float('inf') # Closest forward spacing to any ring vehicle
        d_behind = float('inf') # Closest backward spacing to any ring vehicle
        for v in lane:
            d_fwd = ahead_distance(entry_pos, v.pos, self.C) # Distance from entry to vehicle ahead
            d_bwd = ahead_distance(v.pos, entry_pos, self.C) # Distance from vehicle to entry behind
            d_ahead = min(d_ahead, d_fwd) # Track minimal forward gap
            d_behind = min(d_behind, d_bwd)  # Track minimal rear gap
        return (d_ahead >= min_gap) and (d_behind >= min_gap) # Require sufficient gaps both ahead and behind

    def _attempt_entries(self) -> None:
        for i in range(self.N): # Try to release first-in-queue vehicle for each arm
            q = self.queues[i] # Queue reference
            if not q:
                self.next_needed_headway[i] = 0.0 # No queue means reset needed headway to crit gap logic
                continue # Move to next arm
            lane_idx = q[0].lane_choice # Lane where this queued vehicle intends to merge
            t_next = self._time_to_nearest_ring_vehicle(self.entry_pos[i], lane_idx) # Time until next ring vehicle
            needed = self.next_needed_headway[i] or q[0].crit_gap # Required headway (follow-up if set, else critical)
            if t_next >= needed and self._space_ok_at_merge(lane_idx, self.entry_pos[i], self.cfg.driver.s0 + 2.0):
                v = q.popleft() # Vehicle enters ring from queue head
                v.in_ring = True # Mark as in-ring
                v.pos = (self.entry_pos[i] + 1.0) % self.C # Place slightly downstream of merge point
                v.speed = min(4.0, self.vmax_ring) # Start with small positive speed
                #merge gently, then pick up speed under the IDM—not jump straight to full speed.
                
                v.t_enter_ring = self.t # Record entry time
                delay = self.t - v.t_queue_start # Compute queueing delay
                self.delays_all.append(delay) # Store for global stats
                self.win_delays.append(delay) # Store for window stats
                self.lanes[lane_idx].append(v) # Add to circulating lane
                self.next_needed_headway[i] = v.followup # After first vehicle, use follow-up for platoon
            else:
                if t_next < (self.next_needed_headway[i] or q[0].crit_gap):
                    self.next_needed_headway[i] = 0.0  # If ring vehicle passed, reset to require a new critical gap

    def _leader_at_delayed_time(self, lane: List[Vehicle], pos_delayed: float, snap: Dict[int, Tuple[float, float]]):
        if not snap:
            return None, float('inf'), 0.0 # No snapshot -> treat as free road
        lane_ids = {v.id for v in lane} # IDs of vehicles currently in this lane
        cand = [(p, i) for i, (p, _) in snap.items() if i in lane_ids] # Candidate (pos,id) from snapshot
        #possible leaders at delayed time
        if not cand:
            return None, float('inf'), 0.0 # No leaders found at delayed time
        cand.sort() # Sort by position along ring
        '''
        Here cand is a list of (position, id) pairs from the delayed snapshot. 
        Sorting it by position lets us pick the first car ahead of 
        the ego’s delayed position to use as the leader (and if none is ahead, we wrap to the first item).
        '''
        for p, i in cand:
            if p > pos_delayed: # First vehicle whose delayed position is ahead of our delayed position
                lead_pos, lead_id = p, i
                break
        else:
            lead_pos, lead_id = cand[0] # Wrap-around: leader is the earliest position
        lead_speed = snap[lead_id][1] # Leader's delayed speed
        gap = ahead_distance(pos_delayed, lead_pos, self.C) # Gap at delayed time
        return lead_id, gap, lead_speed # Return leader id, gap, and speed at delayed time

    def _advance_ring(self) -> None:
        dt = self.cfg.dt # Time step
        steps_back = int(round(self.cfg.driver.tau / dt)) # Number of steps corresponding to reaction delay
        snap = self._snapshot(steps_back) if steps_back > 0 else {} # Snapshot from tau seconds ago

        for lane in self.lanes: # Update each circulating lane
            if not lane:
                continue # Skip empty lanes
            lane.sort(key=lambda v: v.pos) # Sort vehicles by current position (increasing along ring)
            next_state: List[Tuple[Vehicle, float, float]] = [] # Collect updates (veh, new_pos, new_speed)
            for v in lane:
                pos_d, v_d = snap.get(v.id, (v.pos, v.speed)) # Delayed ego state; fall back to current if missing
                #ego = current vehicle being updated
                _, gap_d, vL_d = self._leader_at_delayed_time(lane, pos_d, snap) # Leader at delayed time

                drv = self.cfg.driver # Shorthand
                v0 = min(self.vmax_ring, drv.v0_ring) # Effective desired/limit speed
                s0, T, a_max, b, delta = drv.s0, drv.T, drv.a_max, drv.b_comf, drv.delta # IDM params

                if math.isinf(gap_d):
                    gap_d, vL_d = 1e9, v_d # Free road assumption if no leader
                dv = v_d - vL_d # Relative speed (positive if faster than leader)
                s_star = s0 + v_d * T + (v_d * dv) / max(1e-6, 2.0 * math.sqrt(a_max * b)) # IDM desired gap
                s_star = max(s0, s_star) # Enforce minimum static gap
                acc = a_max * (1.0 - (v_d / max(0.1, v0)) ** delta - (s_star / max(1e-3, gap_d)) ** 2) # IDM accel
                v_new = clamp(v.speed + acc * dt, 0.0, v0) # Integrate speed with bounds [0, v0]
                #I.e integration = numerical update from acceleration to speed
                x_new = (v.pos + v_new * dt) % self.C # Advance position modulo circumference
                next_state.append((v, x_new, v_new)) # Stage update to avoid order effects
            for v, x_new, v_new in next_state:
                v.pos, v.speed = x_new, v_new # Commit the staged updates
                '''
                To avoid order effects. If you updated car A right away,
                  car B (processed later) would “see” A’s new position while A “saw” B’s 
                  old one—creating unfair, order-dependent behavior. Staging (aka double buffering) 
                  makes the step synchronous and consistent.
                '''

        # exits: occur when passing own exit; simple proximity check
        for lane in self.lanes:
            survivors: List[Vehicle] = [] # Vehicles that remain after potential exits
            for v in lane:
                exit_pos = self.exit_pos[v.target_exit] # Position of this vehicle's desired exit
                if ahead_distance(v.pos, exit_pos, self.C) < 1.0 and v.t_enter_ring is not None and (self.t - v.t_enter_ring) >= 0.5:
                    self.total_exits += 1 # Count a completed trip
                    self.win_exits += 1 # Count toward this window
                    continue # Vehicle leaves the ring; do not keep
                survivors.append(v) # Keep vehicle if not exiting this step
            lane[:] = survivors # Replace lane list with survivors only

    def _update_queues(self) -> None:
        for i in range(self.N): # For each arm
            qlen = len(self.queues[i]) # Current queue length
            if qlen > self.max_queue_len_global[i]:
                self.max_queue_len_global[i] = qlen # Update all-time per-arm max
            if qlen > self.win_queue_max[i]:
                self.win_queue_max[i] = qlen # Update window per-arm max

    # ------------------------------ reporting --------------------------------
    def _print_header(self) -> None:
        cfg = self.cfg # Alias for brevity
        lam_txt = ", ".join(f"{x:.2f}" for x in cfg.demand.arrivals) # Format arrival rates
        L, T, R = cfg.demand.turning_LTR # Unpack turning ratios
        lane_txt = "single-lane" if cfg.geo.lanes == 1 else "two-lane" # Human-readable lane label
        print(f"SIM: seed={cfg.seed}, horizon={int(cfg.horizon/60)} min, report_every={int(cfg.report_every/60)} min")  # Run header
        print(f"Designs compared with same demand: λ per arm = [{lam_txt}] veh/s") # Demand summary
        print(f"Turning ratios (L/T/R) = [{L:.2f}, {T:.2f}, {R:.2f}]") # Turning ratios summary
        print("-" * 60)
        print(f"=== ROUNDABOUT ({lane_txt}, crit_gap≈{cfg.gaps.crit_gap_mean:.1f}s±{cfg.gaps.crit_gap_sd:.1f}, "
              f"followup≈{cfg.gaps.followup_mean:.1f}s±{cfg.gaps.followup_sd:.1f}) ===") # Model config line

    def _print_window(self, end_time: float) -> None:
        # window delay stats
        if self.win_delays:
            avg_delay = sum(self.win_delays) / len(self.win_delays) # Per-window mean entry delay (s)
            p95 = stats.quantiles(self.win_delays, n=20)[18] if len(self.win_delays) >= 20 else avg_delay # 95th pct approx
        else:
            avg_delay = 0.0  # If no entries, define as zero
            p95 = 0.0 # If no entries, define as zero
        thr_win = (self.win_exits * 3600.0) / self.cfg.report_every # Window throughput scaled to veh/hr
        max_q = f"[{', '.join(str(x) for x in self.win_queue_max)}]" # Pretty per-arm max queue display
        print(f"[{mmss(end_time)}] arrivals={self.win_arrivals}  exits={self.win_exits}  "
              f"throughput={thr_win:.0f} veh/hr  avg_delay={avg_delay:.1f}s  p95={p95:.1f}s  max_q={max_q}") # Window line
        # reset window trackers
        self.win_arrivals = 0  # Reset for next window
        self.win_exits = 0 # Reset for next window
        self.win_delays = [] # Reset for next window
        self.win_queue_max = [0] * self.N # Reset for next window

    def _print_summary(self) -> None:
        # overall delays across hour
        if self.delays_all:
            mean_delay = sum(self.delays_all) / len(self.delays_all)   # Mean entry delay over run
            p95 = stats.quantiles(self.delays_all, n=20)[18] if len(self.delays_all) >= 20 else mean_delay # 95th pct
        else:
            mean_delay = 0.0 # No entries -> zero
            p95 = 0.0 # No entries -> zero
        thr_hour = (self.total_exits * 3600.0) / max(1.0, self.cfg.horizon) # Overall throughput veh/hr
        print("=== HOURLY SUMMARY (00:00–01:00) ===") # Summary header
        print(f"arrivals_total: {self.total_arrivals}") # Total arrivals generated
        print(f"exits_total:    {self.total_exits}") # Total vehicles that completed trip
        print(f"throughput:     {thr_hour:.0f} veh/hr") # Average hourly throughput
        print(f"mean_delay:     {mean_delay:.1f} s/veh") # Mean delay per entering vehicle
        print(f"p95_delay:      {p95:.1f} s") # 95th percentile delay across all entries
        print(f"max_queue_by_arm: [{', '.join(str(x) for x in self.max_queue_len_global)}]") # Peak queues by arm

    # ------------------------------- run -------------------------------------
    def step(self):
        self._push_history()  # Save current ring state for DDE delay reference
        self._spawn_arrivals()  # Generate new arrivals if due at time t
        self._attempt_entries()  # Let front vehicles merge if headway and space allow
        self._advance_ring()  # Advance vehicle states and process exits
        self._update_queues()  # Track max queue lengths
        self.t += self.cfg.dt  # Advance simulation clock by dt

    def run(self) -> None:
        self._print_header()  # Print header describing the scenario
        windows = int(self.cfg.horizon // self.cfg.report_every)  # Number of full report windows in horizon
        window_end = self.cfg.report_every  # End time of current window boundary
        for w in range(windows):  # Loop over windows
            while self.t < window_end:  # Step the sim until we hit the window boundary
                self.step()
            self._print_window(window_end)  # Emit per-window summary line
            window_end += self.cfg.report_every  # Shift to next boundary
        # Cap at exact horizon
        while self.t < self.cfg.horizon:  # If horizon not landed exactly on a boundary, finish up to horizon
            self.step()
        self._print_summary()  # Final hourly summary across the full run


# ------------------------------ CLI ----------------------------------------

def parse_args() -> SimConfig: #ovverrdie coded values from the command line.
    p = argparse.ArgumentParser(description="Roundabout traffic simulation (1‑hour, 5‑min summaries)")  # CLI parser
    p.add_argument("--seed", type=int, default=42)  # RNG seed
    p.add_argument("--horizon", type=float, default=3600.0)  # Total sim time (s)
    p.add_argument("--dt", type=float, default=0.2)  # Time step (s)
    p.add_argument("--report-every", type=float, default=300.0)  # Reporting interval (s)

    p.add_argument("--diameter", type=float, default=45.0)  # Roundabout diameter (m)
    p.add_argument("--lanes", type=int, choices=[1, 2], default=1)  # Circulating lanes
    p.add_argument("--a-lat-max", type=float, default=1.6)  # Lateral accel cap (m/s^2)

    p.add_argument("--arrival", type=float, nargs=4, metavar=("a0","a1","a2","a3"),
                   default=[0.18, 0.12, 0.20, 0.15])  # Four Poisson rates per arm
    p.add_argument("--turning", type=float, nargs=3, metavar=("L","T","R"),
                   default=[0.25, 0.55, 0.20])  # Turning probabilities L/T/R

    p.add_argument("--crit-gap-mean", type=float, default=3.0)  # Critical gap mean (s)
    p.add_argument("--crit-gap-sd", type=float, default=0.6)  # Critical gap SD (s)
    p.add_argument("--followup-mean", type=float, default=2.0)  # Follow-up mean (s)
    p.add_argument("--followup-sd", type=float, default=0.3)  # Follow-up SD (s)

    p.add_argument("--a-max", type=float, default=1.5)  # IDM accel parameter (m/s^2)
    p.add_argument("--b-comf", type=float, default=2.0)  # IDM comfort braking (m/s^2)
    p.add_argument("--T-headway", type=float, default=1.2)  # IDM desired time headway (s)
    p.add_argument("--tau", type=float, default=0.8)  # Reaction delay (s)
    p.add_argument("--v0-ring", type=float, default=12.0)  # Desired free-flow ring speed (m/s)

    args = p.parse_args()  # Parse args from CLI
#The following lines build the config objects from the inputted command-line arguments and
#bundle them into one SimConfig dataclass instance.
    geo = Geometry(diameter=args.diameter, lanes=args.lanes, a_lat_max=args.a_lat_max)  # Build geometry config
    driver = DriverParams(a_max=args.a_max, b_comf=args.b_comf, T=args.T_headway, tau=args.tau, v0_ring=args.v0_ring)  # Driver cfg
    gaps = GapParams(crit_gap_mean=args.crit_gap_mean, crit_gap_sd=args.crit_gap_sd,
                     followup_mean=args.followup_mean, followup_sd=args.followup_sd)  # Gap-acceptance cfg
    demand = Demand(arrivals=args.arrival, turning_LTR=tuple(args.turning))  # Demand cfg

    return SimConfig(seed=args.seed, horizon=args.horizon, dt=args.dt, report_every=args.report_every,
                     geo=geo, driver=driver, gaps=gaps, demand=demand)  # Aggregate config dataclass



def main():
    cfg = parse_args()  # Get config from CLI (CLI = command line interface)
    sim = RoundaboutSim(cfg)  # Instantiate simulator
    sim.run()  # Run the simulation and print summaries


if __name__ == "__main__":  # Execute only when run as a script
    main()  # Entrypoint call