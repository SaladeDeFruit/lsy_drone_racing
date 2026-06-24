# Drone model identification — flight plan

Goal: refine the `cf2x_P250` model offline so the MPC tracks better. The model is
**course-independent**, so this is done with **dedicated excitation flights** (not the race
track, which we don't have until the end). Two tools consume the logs:

- **Linear least squares** for the parametric `so_rpy` constants (mass, thrust mapping,
  attitude coefficients) — these enter the model linearly.
- **`DroneResidualGP`** (`sysid/residual_gp.py`) for the *structural* residual that
  `so_rpy` cannot represent (drag, ground effect, inter-axis coupling).

> **Order matters.** Identify the parametric constants FIRST, then fit the GP on the
> *remaining* residual. Otherwise the GP wastes capacity absorbing a simple constant error
> (e.g. a wrong mass) as a flexible function.

A key identifiability fact: you can only estimate a parameter your data *excites*. Each
flight below is designed to excite a specific part of the model.

---

## What to log (every flight)

Log **raw** at the highest available rate; filter in post-processing.

| Signal | Source | Used for |
| --- | --- | --- |
| `pos` (x,y,z) | Vicon/mocap | state, kinematics |
| `quat` | Vicon/mocap | → `rpy` |
| `vel` | estimator (prefer onboard over differentiated mocap) | translational dynamics |
| `ang_vel` | gyro/estimator | → `drpy` |
| **commands actually sent** (`roll,pitch,yaw,thrust`) | controller output | model input `u` |
| **battery voltage** | telemetry | thrust drifts with battery sag |
| **timestamps** | — | exact `dt` for derivatives |

Post-processing:
- Zero-phase filtering (`scipy.signal.filtfilt`) — no phase lag, only possible offline.
- Derivatives via **Savitzky-Golay** (smooth + differentiate in one pass), or use onboard
  estimates rather than differentiating positions.
- Build one-step transitions `(x_t, u_t, x_{t+1})` at a fixed `dt` matching the MPC step.

---

## Flight 0 — Static (no flight)
- **Weigh the drone on a scale** → fixes `mass` exactly. This decouples `mass` from
  `cmd_f_coef` (confounded in hover), removing the only real parametric identifiability
  trap.

## Flight 1 — Hover + vertical steps/ramps
- **Maneuver:** stable hover, then thrust steps up/down and slow ramps over the full safe
  thrust range. Keep attitude near level.
- **Excites:** vertical acceleration → `cmd_f_coef` (`acc_coef ≈ 0`).
- **Identifies:** thrust mapping via **linear LS** on `vz_dot` vs `thrust`. Residual on
  `vz` also seeds the GP for any thrust nonlinearity / battery sag.

## Flight 2 — Single-axis attitude steps & sweeps
- **Maneuver:** isolated roll steps, then pitch steps, then yaw steps; follow with slow
  frequency sweeps (chirps) on each axis separately, amplitudes up to the racing envelope
  (≈ ±0.5 rad). One axis at a time.
- **Excites:** the per-axis second-order attitude dynamics.
- **Identifies:** `rpy_coef`, `rpy_rates_coef`, `cmd_rpy_coef` (9 scalars) by **linear LS**
  on `ddrpy = rpy_coef·rpy + rpy_rates_coef·drpy + cmd_rpy_coef·cmd_rpy`. This is the core
  of an attitude-interface MPC and the cheapest big accuracy win.

## Flight 3 — Combined aggressive maneuvers (multi-axis)
- **Maneuver:** simultaneous roll+pitch+yaw rates, fast figure-eights, coordinated turns —
  high angular rates on several axes at once.
- **Excites:** inter-axis **gyroscopic coupling** `ω×Jω` that `so_rpy` (decoupled per axis)
  cannot represent.
- **Identifies:** GP residual on `drpy` channels, with features `feature_dims = (rpy,
  drpy, cmd_rpy)`. This is the main blind spot of `so_rpy` for racing.

## Flight 4 — High-speed straight passes
- **Maneuver:** straight accelerate/decelerate runs at a range of speeds (slow → max), in
  +x, +y and diagonal directions.
- **Excites:** velocity-dependent **drag**.
- **Identifies:** GP residual on `vel` channels, features `feature_dims = (vel, rpy)`. If
  the residual is cleanly `∝ -v` (or `-|v|v`), consider distilling it into an analytic drag
  term and switching the controller model to `so_rpy_rotor_drag` (which already has a
  `drag_matrix`).

## Flight 5 — Low-altitude passes
- **Maneuver:** level passes at several constant low heights (close to the floor → up).
- **Excites:** **ground effect** (extra lift near the ground).
- **Identifies:** GP residual on `vz`, features including height `z` (index 2). Absent from
  every `so_rpy` variant → GP-only.

---

## Coverage & validation
- **Cover the racing envelope.** The GP only knows what it has seen; high-speed residual
  requires high-speed data (Flight 4). Don't expect good corrections outside the flown
  envelope — watch the GP std as an out-of-distribution flag.
- **Hold out** ~20% of each flight for validation; check `corrected_rmse < nominal_rmse`
  (see `DroneResidualGP.evaluate`) before trusting the refined model.
- **Repeat** flights for statistics and to average sensor noise.
- **Safety:** stay inside a known safe volume, ramp amplitudes gradually, keep an operator
  kill-switch.

## Feeding the pipeline
```python
from sysid.residual_gp import DroneResidualGP
gp = DroneResidualGP(dt=1/freq, feature_dims=(...per flight...), output_dims=(...))
gp.fit(X_t, U_t, X_tp1)          # X_t,U_t,X_tp1 from the filtered logs
print(gp.evaluate(X_te, U_te, X_tp1_te))   # nominal vs corrected RMSE
```
Iterate: model₁ (nominal) → fly → identify → model₂ → fly → … (see project notes).
