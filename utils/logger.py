"""
logger.py
---------
Lightweight experiment logger for AlphaPortfolio.

Outputs per run:
  runs/<experiment_name>_<timestamp>/
      config.json      — hyperparameter snapshot
      metrics.jsonl    — one JSON object per log() call
      checkpoints/     — .pt checkpoint directory

Usage
-----
>>> logger = ExperimentLogger()
>>> logger.log_hyperparams()
>>> stats = episode_tracker.end_episode(total_return, steps)
>>> logger.log_episode(stats)
>>> torch.save(agent.state_dict(), logger.checkpoint_path(episode=100))
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import CFG


# ──────────────────────────────────────────────────────────────────────
# Metric accumulator
# ──────────────────────────────────────────────────────────────────────

class MetricBuffer:
    """Accumulates scalar values within a single episode / eval window."""

    def __init__(self) -> None:
        self._data: Dict[str, List[float]] = defaultdict(list)

    def add(self, key: str, value: float) -> None:
        self._data[key].append(float(value))

    def mean(self, key: str) -> float:
        vals = self._data.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    def last(self, key: str) -> Optional[float]:
        vals = self._data.get(key, [])
        return vals[-1] if vals else None

    def clear(self) -> None:
        self._data.clear()

    def to_dict(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self._data.items()}


# ──────────────────────────────────────────────────────────────────────
# Per-episode tracker
# ──────────────────────────────────────────────────────────────────────

class EpisodeTracker:
    """
    Call start_episode() / log_step() / end_episode() from the training loop.
    end_episode() returns a stats dict ready to pass to ExperimentLogger.log_episode().
    """

    def __init__(self) -> None:
        self.episode: int = 0
        self.total_steps: int = 0
        self.episode_returns: List[float] = []
        self._step_buf = MetricBuffer()
        self._ep_start: float = 0.0

    def start_episode(self) -> None:
        self._step_buf.clear()
        self._ep_start = time.time()

    def log_step(self, reward: float, **extra: float) -> None:
        """Log one environment step's reward and any extra scalar metrics."""
        self._step_buf.add("reward", reward)
        for k, v in extra.items():
            self._step_buf.add(k, v)
        self.total_steps += 1

    def end_episode(self, episode_return: float, episode_length: int) -> Dict[str, Any]:
        """Finalise episode; returns a flat stats dict."""
        self.episode += 1
        self.episode_returns.append(episode_return)
        elapsed = time.time() - self._ep_start

        stats: Dict[str, Any] = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "elapsed_s": round(elapsed, 3),
            "mean_return_10":  float(np.mean(self.episode_returns[-10:])),
            "mean_return_100": float(np.mean(self.episode_returns[-100:])),
        }
        stats.update(self._step_buf.to_dict())
        return stats


# ──────────────────────────────────────────────────────────────────────
# Experiment logger
# ──────────────────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    File-based experiment logger.

    Parameters
    ----------
    log_dir         : root directory for all runs (default: CFG.LOG_DIR)
    experiment_name : prefix for this run's directory (default: CFG.EXPERIMENT_NAME)
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = experiment_name or CFG.EXPERIMENT_NAME
        self.run_dir = Path(log_dir or CFG.LOG_DIR) / f"{name}_{timestamp}"
        self.checkpoint_dir = self.run_dir / "checkpoints"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._wall_start = time.time()
        self._best_return: float = -np.inf

        print(f"[Logger] Run directory: {self.run_dir}")

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Append a metrics dict as a single line to metrics.jsonl."""
        record: Dict[str, Any] = {"wall_time": round(time.time() - self._wall_start, 3)}
        if step is not None:
            record["step"] = step
        record.update(_sanitise(metrics))
        with open(self._metrics_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def log_hyperparams(self, cfg=None) -> None:
        """Snapshot a config dataclass to config.json."""
        import dataclasses
        cfg = cfg or CFG
        cfg_dict = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)
        path = self.run_dir / "config.json"
        with open(path, "w") as fh:
            json.dump(cfg_dict, fh, indent=2)
        print(f"[Logger] Config saved to {path}")

    def log_episode(self, stats: Dict[str, Any]) -> None:
        """Log episode stats to disk and print a one-line summary."""
        self.log(stats, step=stats.get("episode"))
        ep = stats.get("episode", "?")
        ret = stats.get("episode_return", 0.0)
        mean_ret = stats.get("mean_return_100", 0.0)
        steps = stats.get("total_steps", 0)
        print(
            f"[Ep {ep:>5}]  return={ret:+.5f}  "
            f"mean_100={mean_ret:+.5f}  steps={steps:,}"
        )

    # ------------------------------------------------------------------
    # Checkpoint paths
    # ------------------------------------------------------------------

    def checkpoint_path(self, episode: int, tag: str = "agent") -> Path:
        """Path for a periodic checkpoint."""
        return self.checkpoint_dir / f"{tag}_ep{episode:06d}.pt"

    def best_checkpoint_path(self, tag: str = "agent") -> Path:
        """Path for the best-so-far checkpoint (overwritten each time)."""
        return self.checkpoint_dir / f"{tag}_best.pt"

    def is_new_best(self, episode_return: float) -> bool:
        """Returns True and updates the high-water mark if this is a new best."""
        if episode_return > self._best_return:
            self._best_return = episode_return
            return True
        return False

    # ------------------------------------------------------------------
    # Read-back
    # ------------------------------------------------------------------

    def load_metrics(self) -> List[Dict]:
        """Read all metrics from the JSONL file into a list of dicts."""
        if not self._metrics_path.exists():
            return []
        with open(self._metrics_path) as fh:
            return [json.loads(line) for line in fh if line.strip()]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _sanitise(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy scalars / arrays to plain Python types for JSON serialisation."""
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out
