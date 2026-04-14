"""
evaluate.py
-----------
Evaluation metrics and visualisation for AlphaPortfolio.

Metrics
-------
  cumulative_return  : (final_value / initial_value) - 1
  sharpe_ratio       : annualised (mean_excess / std_excess) * sqrt(252)
  max_drawdown       : worst peak-to-trough percentage decline
  win_rate           : fraction of steps with positive return

Comparison
----------
  compare_strategies(strategies, env)
      Runs each strategy for n_episodes, aggregates metrics, returns a
      DataFrame-like dict and optionally generates matplotlib plots.

Plots saved to experiments/results/ by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt
import numpy as np

from config import CFG
from environment.market_env import MarketEnv
from orchestrator.portfolio import Portfolio

RESULTS_DIR = Path("experiments") / "results"
_REGIME_WINDOW = 20


# ──────────────────────────────────────────────────────────────────────
# Financial metrics
# ──────────────────────────────────────────────────────────────────────

def cumulative_return(portfolio_values: np.ndarray) -> float:
    """Total return over the episode: (V_T / V_0) - 1."""
    pv = np.asarray(portfolio_values, dtype=np.float64)
    if len(pv) < 2 or pv[0] == 0:
        return 0.0
    return float(pv[-1] / pv[0]) - 1.0


def sharpe_ratio(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    daily_returns   : step-by-step percentage returns (not cumulative)
    risk_free_rate  : annual risk-free rate (default 0)
    periods_per_year: trading days per year (default 252)
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    excess = r - daily_rf
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Largest peak-to-trough drawdown as a negative fraction.
    e.g. -0.25 means a 25% drawdown from peak.
    """
    pv = np.asarray(portfolio_values, dtype=np.float64)
    if len(pv) < 2:
        return 0.0
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / (peak + 1e-10)
    return float(dd.min())


def win_rate(daily_returns: np.ndarray) -> float:
    """Fraction of steps with a strictly positive return."""
    r = np.asarray(daily_returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    return float(np.mean(r > 0))


def compute_metrics(
    portfolio_values: np.ndarray,
    daily_returns: np.ndarray,
) -> Dict[str, float]:
    """Compute all four metrics from a single episode trajectory."""
    return {
        "cumulative_return": cumulative_return(portfolio_values),
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "win_rate": win_rate(daily_returns),
        "final_value": float(portfolio_values[-1]) if len(portfolio_values) else 0.0,
        "n_steps": len(daily_returns),
    }


def _average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metric dicts across multiple episodes."""
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


# ──────────────────────────────────────────────────────────────────────
# Episode runners
# ──────────────────────────────────────────────────────────────────────

def _recent_prices(env: MarketEnv, window: int = _REGIME_WINDOW) -> np.ndarray:
    t = env._t
    return env.close_prices[max(0, t - window) : t + 1]


def run_episode_agent(
    agent, env: MarketEnv
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one episode for any agent with a select_action(obs) method.

    Returns
    -------
    portfolio_values : shape (T+1,) — value at each step including initial
    daily_returns    : shape (T,)   — step-by-step pct returns
    """
    if hasattr(agent, "reset"):
        agent.reset()
    obs, info = env.reset()
    portfolio_values = [info["portfolio_value"]]
    daily_returns: List[float] = []
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        portfolio_values.append(info["portfolio_value"])
        daily_returns.append(info["pct_return"])

    return np.array(portfolio_values), np.array(daily_returns)


def run_episode_portfolio(
    portfolio: Portfolio, env: MarketEnv
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one episode for the full Portfolio orchestrator.
    """
    obs, info = env.reset()
    portfolio_values = [info["portfolio_value"]]
    daily_returns: List[float] = []
    done = False

    while not done:
        recent_px = _recent_prices(env)
        action, _ = portfolio.select_action(obs, recent_px)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        portfolio_values.append(info["portfolio_value"])
        daily_returns.append(info["pct_return"])

    return np.array(portfolio_values), np.array(daily_returns)


# ──────────────────────────────────────────────────────────────────────
# Evaluation functions
# ──────────────────────────────────────────────────────────────────────

def evaluate_agent(
    agent,
    env: MarketEnv,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """
    Evaluate a simple agent (RandomAgent, BuyAndHoldAgent, DQNAgent, etc.)
    over n_episodes.  Returns averaged metrics.
    """
    if hasattr(agent, "eval_mode"):
        agent.eval_mode()

    all_metrics = []
    for _ in range(n_episodes):
        pv, dr = run_episode_agent(agent, env)
        all_metrics.append(compute_metrics(pv, dr))

    if hasattr(agent, "train_mode"):
        agent.train_mode()

    return _average_metrics(all_metrics)


def evaluate_portfolio(
    portfolio: Portfolio,
    env: MarketEnv,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate the full Portfolio orchestrator over n_episodes."""
    portfolio.eval_mode()
    all_metrics = []
    for _ in range(n_episodes):
        pv, dr = run_episode_portfolio(portfolio, env)
        all_metrics.append(compute_metrics(pv, dr))
    portfolio.train_mode()
    return _average_metrics(all_metrics)


def compare_strategies(
    strategies: Dict[str, Any],
    env: MarketEnv,
    n_episodes: int = 10,
    save_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple strategies and optionally save comparison plots.

    Parameters
    ----------
    strategies : {name: agent_or_portfolio}
                 Portfolio instances are detected automatically.
    env        : evaluation environment (use test split)
    n_episodes : episodes per strategy
    save_dir   : directory for saved plots and JSON; defaults to RESULTS_DIR

    Returns
    -------
    {strategy_name: metrics_dict}
    """
    save_dir = Path(save_dir) if save_dir else RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}
    for name, strategy in strategies.items():
        print(f"  Evaluating {name}...", end=" ", flush=True)
        if isinstance(strategy, Portfolio):
            metrics = evaluate_portfolio(strategy, env, n_episodes)
        else:
            metrics = evaluate_agent(strategy, env, n_episodes)
        results[name] = metrics
        cr = metrics["cumulative_return"]
        sr = metrics["sharpe_ratio"]
        md = metrics["max_drawdown"]
        print(f"ret={cr:+.4f}  sharpe={sr:.3f}  mdd={md:.4f}")

    save_results(results, save_dir)
    plot_metrics_comparison(results, save_dir / "metrics_comparison.png")

    return results


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    episode_stats_dict: Dict[str, List[Dict]],
    save_path: Optional[Union[str, Path]] = None,
    smooth_window: int = 20,
) -> None:
    """
    Plot smoothed episode returns for one or more training runs.

    Parameters
    ----------
    episode_stats_dict : {label: [episode_stats, ...]}
                         Each inner list comes from train_specialists() or
                         train_orchestrator().
    save_path          : file path to save figure (PNG)
    smooth_window      : rolling-mean window for smoothing
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, stats_list in episode_stats_dict.items():
        returns = np.array([s["episode_return"] for s in stats_list])
        episodes = np.arange(1, len(returns) + 1)

        # Raw (faint) + smoothed
        axes[0].plot(episodes, returns, alpha=0.2)
        if len(returns) >= smooth_window:
            smoothed = np.convolve(
                returns, np.ones(smooth_window) / smooth_window, mode="valid"
            )
            axes[0].plot(
                episodes[smooth_window - 1 :], smoothed, label=label, linewidth=1.8
            )
        else:
            axes[0].plot(episodes, returns, label=label, linewidth=1.8)

        # Cumulative mean
        cum_mean = np.cumsum(returns) / (np.arange(len(returns)) + 1)
        axes[1].plot(episodes, cum_mean, label=label, linewidth=1.8)

    axes[0].set_title("Episode Return (smoothed)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Cumulative Mean Return")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Mean Return")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_portfolio_comparison(
    trajectory_dict: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot normalised portfolio value over a single test episode.

    Parameters
    ----------
    trajectory_dict : {strategy_name: portfolio_values_array}
    save_path       : file path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, pv in trajectory_dict.items():
        pv_norm = np.asarray(pv) / pv[0]          # normalise to 1.0 at start
        ax.plot(pv_norm, label=name, linewidth=1.8)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Normalised Portfolio Value — Test Episode")
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio Value (normalised)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Bar chart comparing key metrics across strategies.
    """
    metrics_to_plot = ["cumulative_return", "sharpe_ratio", "max_drawdown", "win_rate"]
    n_metrics = len(metrics_to_plot)
    n_strategies = len(results)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    names = list(results.keys())
    x = np.arange(n_strategies)
    colours = plt.cm.tab10(np.linspace(0, 1, n_strategies))

    for ax, metric in zip(axes, metrics_to_plot):
        values = [results[n].get(metric, 0.0) for n in names]
        bars = ax.bar(x, values, color=colours)
        ax.set_title(metric.replace("_", "\n"), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace("_", "\n") for n in names], fontsize=7, rotation=0
        )
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle("Strategy Comparison — Test Split", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────

def save_results(
    results: Dict[str, Any],
    save_dir: Union[str, Path],
    filename: str = "metrics.json",
) -> Path:
    """Serialize results dict to JSON."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename

    # Convert numpy scalars to Python floats
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as fh:
        json.dump(_clean(results), fh, indent=2)
    print(f"[Evaluate] Results saved to {path}")
    return path


def _save_or_show(fig: plt.Figure, path: Optional[Union[str, Path]]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Evaluate] Plot saved to {path}")
    plt.close(fig)
