"""Schema-agnostic flight logger for offline system identification.

Records, at every control step, the timestamp, the full observation dict (whatever keys it
contains), the action that was sent, and optional extra fields. At the end of the flight it
dumps everything to disk as a ``.npz`` (structured, one array per obs key) plus a flattened
``.csv`` for quick inspection. The exact log content does not need to be known in advance —
we study the file later and build the ``(x_t, u_t, x_{t+1})`` transitions for the GP then.

Typical use from a controller:

    self._logger = FlightLogger("logs", run_name="roll_sweep")
    # in compute_control, before mutating obs:
    self._logger.log(t=self._tick / freq, obs=obs, action=action)
    # in episode_callback:
    self._logger.save()
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np


class FlightLogger:
    """Buffers per-step records and writes them to ``.npz`` + ``.csv``."""

    def __init__(self, out_dir: str | Path = "logs", run_name: str | None = None):
        """Initialize the logger.

        Args:
            out_dir: Directory where log files are written (created if missing).
            run_name: Optional label included in the filename. A timestamp is always added.
        """
        self._dir = Path(out_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{run_name}_{stamp}" if run_name else stamp
        self._stem = self._dir / f"flight_{name}"
        self._t: list[float] = []
        self._obs: list[dict[str, np.ndarray]] = []
        self._act: list[np.ndarray] = []
        self._extra: list[dict[str, Any]] = []
        self._keys: list[str] | None = None

    @property
    def n_steps(self) -> int:
        """Number of recorded steps."""
        return len(self._t)

    def log(
        self,
        t: float,
        obs: dict[str, Any],
        action: Any,
        extra: dict[str, Any] | None = None,
    ):
        """Record one step. Copies obs so later in-place mutation does not corrupt the log.

        Args:
            t: Timestamp of this step, in seconds.
            obs: Observation dict (any keys; values array-like or scalar).
            action: Action sent to the environment (array-like).
            extra: Optional extra scalars to store (e.g. battery voltage, reward).
        """
        snap = {k: np.array(v, copy=True) for k, v in obs.items()}
        if self._keys is None:
            self._keys = list(snap.keys())
        elif list(snap.keys()) != self._keys:
            # Keep going but warn: identification post-processing assumes a stable schema.
            missing = set(self._keys) ^ set(snap.keys())
            print(f"[FlightLogger] warning: obs keys changed ({missing}) at t={t:.3f}")
        self._t.append(float(t))
        self._obs.append(snap)
        self._act.append(np.asarray(action, dtype=float).ravel())
        self._extra.append(dict(extra) if extra else {})

    def _stack(self) -> dict[str, np.ndarray]:
        """Stack the buffered records into one array per field."""
        out: dict[str, np.ndarray] = {"t": np.asarray(self._t, dtype=float)}
        for k in self._keys or []:
            out[f"obs_{k}"] = np.stack([rec[k] for rec in self._obs])
        out["action"] = np.stack(self._act)
        extra_keys = sorted({k for e in self._extra for k in e})
        for k in extra_keys:
            out[f"extra_{k}"] = np.asarray(
                [e.get(k, np.nan) for e in self._extra], dtype=float
            )
        return out

    def save(self) -> Path:
        """Write the buffered log to ``<stem>.npz`` and ``<stem>.csv``. Returns the npz path."""
        if self.n_steps == 0:
            print("[FlightLogger] nothing to save (0 steps).")
            return self._stem.with_suffix(".npz")
        data = self._stack()
        npz_path = self._stem.with_suffix(".npz")
        np.savez_compressed(npz_path, **data)

        # Flattened CSV for quick inspection (one column per scalar component).
        columns: list[str] = []
        flat_cols: list[np.ndarray] = []
        for key, arr in data.items():
            a = arr.reshape(arr.shape[0], -1)
            if a.shape[1] == 1:
                columns.append(key)
                flat_cols.append(a[:, 0])
            else:
                for j in range(a.shape[1]):
                    columns.append(f"{key}_{j}")
                    flat_cols.append(a[:, j])
        matrix = np.column_stack(flat_cols)
        csv_path = self._stem.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            w.writerows(matrix.tolist())

        print(f"[FlightLogger] saved {self.n_steps} steps -> {npz_path.name}, {csv_path.name}")
        return npz_path
