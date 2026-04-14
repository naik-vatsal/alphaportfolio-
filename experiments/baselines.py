"""
baselines.py
------------
Simple benchmark strategies for AlphaPortfolio comparisons.

RandomAgent
    Selects a uniformly random action for each stock at every step.
    Lower bound; should be beaten convincingly by any trained agent.

BuyAndHoldAgent
    Buys all stocks equally on the first step of each episode, then holds
    for the remainder.  Classic passive benchmark.

EqualWeightRebalanceAgent
    Rebalances to equal weights every `rebalance_every` steps.
    Represents a simple systematic strategy without any learning.

All baseline agents implement the same interface as BaseAgent:
    select_action(obs) → np.ndarray
    eval_mode() / train_mode() → no-op
    reset() → resets internal episode state

Usage
-----
    from experiments.baselines import RandomAgent, BuyAndHoldAgent
    from training.evaluate import evaluate_agent

    rng_metrics = evaluate_agent(RandomAgent(n_stocks=5), test_env)
    bnh_metrics = evaluate_agent(BuyAndHoldAgent(n_stocks=5), test_env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from config import CFG
from environment.market_env import MarketEnv
from training.evaluate import evaluate_agent, save_results, RESULTS_DIR


# ──────────────────────────────────────────────────────────────────────
# Baseline agents
# ──────────────────────────────────────────────────────────────────────

class RandomAgent:
    """
    Uniformly random portfolio actions.

    Serves as a statistical lower bound.  Any trained strategy should
    produce a higher Sharpe ratio and cumulative return than this.
    """

    name = "random"

    def __init__(self, n_stocks: int = CFG.N_STOCKS, seed: Optional[int] = None) -> None:
        self.n_stocks = n_stocks
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self._rng.integers(0, 3, size=self.n_stocks, dtype=np.int64)

    def reset(self) -> None:
        pass   # stateless between episodes

    def eval_mode(self) -> None:
        pass

    def train_mode(self) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {"agent": self.name}

    def __repr__(self) -> str:
        return f"RandomAgent(n_stocks={self.n_stocks})"


class BuyAndHoldAgent:
    """
    Equal-weight buy on step 0, hold for the rest of the episode.

    Models a passive investor who allocates equally to all assets at the
    start and ignores all subsequent signals.
    """

    name = "buy_and_hold"

    def __init__(self, n_stocks: int = CFG.N_STOCKS) -> None:
        self.n_stocks = n_stocks
        self._first_step: bool = True

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        if self._first_step:
            self._first_step = False
            return np.full(self.n_stocks, 2, dtype=np.int64)   # BUY all
        return np.ones(self.n_stocks, dtype=np.int64)           # HOLD all

    def reset(self) -> None:
        """Must be called between episodes (evaluate.run_episode_agent calls this)."""
        self._first_step = True

    def eval_mode(self) -> None:
        pass

    def train_mode(self) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {"agent": self.name}

    def __repr__(self) -> str:
        return f"BuyAndHoldAgent(n_stocks={self.n_stocks})"


class EqualWeightRebalanceAgent:
    """
    Rebalances to equal weights every `rebalance_every` steps.

    On rebalance days: issues BUY for all stocks (effectively a target
    equal-weight allocation via the env's cash-split logic).
    On off days: issues HOLD.
    """

    name = "equal_weight_rebalance"

    def __init__(
        self, n_stocks: int = CFG.N_STOCKS, rebalance_every: int = 21
    ) -> None:
        self.n_stocks = n_stocks
        self.rebalance_every = rebalance_every
        self._step: int = 0

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        action = (
            np.full(self.n_stocks, 2, dtype=np.int64)    # BUY on rebalance
            if self._step % self.rebalance_every == 0
            else np.ones(self.n_stocks, dtype=np.int64)  # HOLD otherwise
        )
        self._step += 1
        return action

    def reset(self) -> None:
        self._step = 0

    def eval_mode(self) -> None:
        pass

    def train_mode(self) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {"agent": self.name, "rebalance_every": self.rebalance_every}

    def __repr__(self) -> str:
        return f"EqualWeightRebalanceAgent(rebalance_every={self.rebalance_every})"


# ──────────────────────────────────────────────────────────────────────
# Convenience evaluator
# ──────────────────────────────────────────────────────────────────────

def evaluate_baselines(
    env: MarketEnv,
    n_episodes: int = 10,
    save_dir=None,
) -> Dict[str, Dict[str, float]]:
    """
    Run all three baselines on `env` and return {name: metrics}.

    Parameters
    ----------
    env        : evaluation MarketEnv (typically test split)
    n_episodes : episodes per baseline
    save_dir   : optional directory to save metrics.json and plots

    Returns
    -------
    {agent_name: metrics_dict}
    """
    n = env.n_stocks
    baselines = {
        "random":                RandomAgent(n_stocks=n),
        "buy_and_hold":          BuyAndHoldAgent(n_stocks=n),
        "equal_weight_rebalance": EqualWeightRebalanceAgent(n_stocks=n),
    }

    results: Dict[str, Dict[str, float]] = {}
    print("\n[Baselines] Evaluating benchmarks...")
    for name, agent in baselines.items():
        metrics = evaluate_agent(agent, env, n_episodes=n_episodes)
        results[name] = metrics
        print(
            f"  {name:<28}  "
            f"ret={metrics['cumulative_return']:+.4f}  "
            f"sharpe={metrics['sharpe_ratio']:.3f}  "
            f"mdd={metrics['max_drawdown']:.4f}  "
            f"win={metrics['win_rate']:.3f}"
        )

    if save_dir is not None:
        save_results(results, save_dir, filename="baseline_metrics.json")

    return results
