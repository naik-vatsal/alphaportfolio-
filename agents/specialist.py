"""
specialist.py
-------------
Named DQN specialists for AlphaPortfolio.

Each specialist is a DQNAgent subclass with:
  - A REGIME class attribute that the orchestrator uses to route data.
  - Regime-tuned default hyperparameters (overridable via cfg).
  - A describe() method for logging and reporting.

Specialists
-----------
MomentumAgent
    Trained on trending / directional market segments.
    Prefers longer horizons → higher gamma.

MeanReversionAgent
    Trained on high-volatility, mean-reverting market segments.
    Prefers shorter horizons → lower gamma, higher exploration.

MacroAgent
    Trained on the full market history (no regime filter).
    Serves as the baseline / ensemble anchor.

Routing contract (enforced by the orchestrator, not here)
---------------------------------------------------------
Each specialist exposes `REGIME` and `segment_filter(features, prices)`
which the orchestrator calls to decide whether a window is suitable for
this specialist.  The method returns a bool; returning True means "I
should train on this window".  Specialists do NOT perform the routing
themselves — they only declare the heuristic.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from config import CFG
from agents.dqn_agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────
# Momentum Specialist
# ──────────────────────────────────────────────────────────────────────

class MomentumAgent(DQNAgent):
    """
    DQN specialist for trending / momentum market regimes.

    Regime heuristic
    ----------------
    A window is considered trending when the mean absolute 21-day rolling
    return (across all stocks) exceeds a threshold, indicating persistent
    directional movement.

    Hyperparameter adjustments vs default
    --------------------------------------
    - Higher gamma: trends unfold over many days, so future rewards matter more.
    - Lower epsilon end: converges to a more decisive policy.
    """

    REGIME = "momentum"

    def __init__(self, obs_dim: int, n_stocks: int, cfg=CFG) -> None:
        # Regime-specific overrides (non-destructive copies)
        super().__init__(
            obs_dim=obs_dim,
            n_stocks=n_stocks,
            cfg=cfg,
            name="momentum_agent",
        )
        # Override after super().__init__ so cfg singleton is not mutated
        self.cfg = _CfgOverride(cfg, GAMMA=0.995, EPSILON_END=0.02)
        # Re-point the optimizer to the same lr (gamma doesn't affect optimizer)

    def segment_filter(
        self, features: np.ndarray, prices: np.ndarray, threshold: float = 0.6
    ) -> bool:
        """
        Returns True if the price window shows strong directional momentum.

        Heuristic: mean absolute 21-day return across stocks exceeds `threshold`
        standard deviations above the feature-set mean.
        """
        if len(prices) < 22:
            return False
        ret_21 = (prices[-1] - prices[-21]) / (prices[-21] + 1e-8)  # (n_stocks,)
        score = float(np.mean(np.abs(ret_21)))
        return score > threshold

    def describe(self) -> str:
        return (
            f"MomentumAgent | regime=trending | "
            f"gamma=0.995 | eps_end=0.02 | steps={self.total_steps}"
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["regime"] = self.REGIME
        return stats


# ──────────────────────────────────────────────────────────────────────
# Mean-Reversion Specialist
# ──────────────────────────────────────────────────────────────────────

class MeanReversionAgent(DQNAgent):
    """
    DQN specialist for high-volatility, mean-reverting market regimes.

    Regime heuristic
    ----------------
    A window is considered mean-reverting when the cross-sectional
    volatility (std of daily returns across all stocks) is elevated,
    suggesting oscillatory price action where oversold/overbought signals
    are more reliable.

    Hyperparameter adjustments vs default
    --------------------------------------
    - Lower gamma: mean reversion plays out quickly; near-term rewards dominate.
    - Higher epsilon end: more exploration to avoid committing to stale signals.
    - Larger hidden dim: wider net to capture non-linear reversion patterns.
    """

    REGIME = "mean_reversion"

    def __init__(self, obs_dim: int, n_stocks: int, cfg=CFG) -> None:
        super().__init__(
            obs_dim=obs_dim,
            n_stocks=n_stocks,
            cfg=cfg,
            name="mean_reversion_agent",
        )
        self.cfg = _CfgOverride(cfg, GAMMA=0.97, EPSILON_END=0.08, HIDDEN_DIM=512)

    def segment_filter(
        self, features: np.ndarray, prices: np.ndarray, threshold: float = 1.5
    ) -> bool:
        """
        Returns True if the window exhibits elevated cross-sectional volatility.

        Heuristic: std of daily returns across stocks is `threshold`× the
        rolling mean std over the full features window.
        """
        if len(prices) < 10:
            return False
        daily_ret = np.diff(prices, axis=0) / (prices[:-1] + 1e-8)  # (T-1, n_stocks)
        recent_vol = float(np.std(daily_ret[-5:]))
        baseline_vol = float(np.std(daily_ret)) + 1e-8
        return (recent_vol / baseline_vol) > threshold

    def describe(self) -> str:
        return (
            f"MeanReversionAgent | regime=high_vol | "
            f"gamma=0.97 | eps_end=0.08 | steps={self.total_steps}"
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["regime"] = self.REGIME
        return stats


# ──────────────────────────────────────────────────────────────────────
# Macro Specialist (full-market baseline)
# ──────────────────────────────────────────────────────────────────────

class MacroAgent(DQNAgent):
    """
    DQN specialist trained on all market conditions without regime filtering.

    This agent acts as the ensemble anchor: it is always eligible for
    training and provides a stable baseline that the orchestrator can fall
    back on when no specialist regime is confidently detected.

    Uses default CFG hyperparameters without modification.
    """

    REGIME = "macro"

    def __init__(self, obs_dim: int, n_stocks: int, cfg=CFG) -> None:
        super().__init__(
            obs_dim=obs_dim,
            n_stocks=n_stocks,
            cfg=cfg,
            name="macro_agent",
        )

    def segment_filter(
        self, features: np.ndarray, prices: np.ndarray, **kwargs
    ) -> bool:
        """MacroAgent trains on every window — always returns True."""
        return True

    def describe(self) -> str:
        return (
            f"MacroAgent | regime=all_conditions | "
            f"gamma={self.cfg.GAMMA} | steps={self.total_steps}"
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["regime"] = self.REGIME
        return stats


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

class _CfgOverride:
    """
    Lightweight proxy that wraps CFG and overrides specific attributes.
    Avoids mutating the global singleton so all specialists can coexist.
    """

    def __init__(self, base_cfg, **overrides) -> None:
        self._base = base_cfg
        self._overrides = overrides

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)

    def __repr__(self) -> str:
        return f"_CfgOverride({self._overrides})"
