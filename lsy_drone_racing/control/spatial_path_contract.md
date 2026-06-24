# Stage 1 → TOPP-RA contract (spatial path)

The 4-stage pipeline is: **spatial path (1) → TOPP-RA (2) → TrajectoryGenerator sampler (3)
→ MPC (4)**. Stages 3 and 4 are implemented (`trajectory_generator_test.py`,
`attitude_mpc_racing.py`). This note specifies what **stage 1** must provide so TOPP-RA can
consume it.

## What the MPC expects from stage 1

`AttitudeMPCRacing._build_reference` calls:

```python
from lsy_drone_racing.control.spatial_path import build_spatial_path
path = build_spatial_path(obs, config)
```

So stage 1 must expose a module `lsy_drone_racing/control/spatial_path.py` with:

```python
def build_spatial_path(obs: dict, config) -> "toppra GeometricPath":
    ...
```

(If you name the module/function differently, update the import in `_build_reference`.)

## What `path` must be

A **toppra-compatible geometric path** — i.e. a `toppra.SplineInterpolator` (or any object
implementing toppra's `GeometricPath` interface). Concretely:

- Built from the gate/obstacle-aware **waypoints** — **stage 1 is the ONLY place that
  computes waypoints / handles gates / obstacles**. Nothing downstream knows about geometry.
- `dof == 3` (x, y, z). Yaw is **not** part of the path (yaw is free in the MPC).
- Parameterized over a normalized arc parameter, e.g.:

  ```python
  import numpy as np, toppra as ta
  ss = np.linspace(0.0, 1.0, len(waypoints))      # normalized arc
  path = ta.SplineInterpolator(ss, waypoints, bc_type="clamped")  # start/end at rest
  return path
  ```

- Use `bc_type="clamped"` if you want zero velocity at the endpoints (typical for a race
  start/finish).

## What TOPP-RA (stage 2, inside the MPC) does with it

```python
pc_vel = JointVelocityConstraint(vel_limit)     # (3,2) per-axis [lower, upper]
pc_acc = JointAccelerationConstraint(acc_limit) # (3,2); z asymmetric, e.g. [-8.2, 4.5]
instance = TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
traj = instance.compute_trajectory()            # callable traj(t,0/1/2), traj.duration
```

The kinematic limits are configurable per axis and asymmetric. Defaults live in
`attitude_mpc_racing.DEFAULT_VEL_LIMIT` / `DEFAULT_ACC_LIMIT`; override them via a
`[toppra]` section in the config:

```toml
[toppra]
vel_limit = [[-1.8, 1.8], [-1.8, 1.8], [-1.0, 1.0]]
acc_limit = [[-4.5, 4.5], [-4.5, 4.5], [-8.2, 4.5]]   # z: gravity helps deceleration
```

`traj` is then handed to `TrajectoryGenerator(traj, freq)`, which samples it into the
reference arrays the MPC tracks. Stage 1 never samples and never sees the MPC.
