"""
run_all.py
----------
Five experiment configurations for AlphaPortfolio.

Experiments
-----------
1  baseline_dqn            Single MacroAgent, no orchestration
2  dqn_ucb                 Portfolio with UCB bandit, no reward sharing
3  dqn_ucb_coord           Portfolio + UCB + coordinator reward sharing
4  full_system             Complete system including LLM regime detection
5  ablation_alpha          Sweep COORD_ALPHA over {0.0, 0.7, 1.0}

Each experiment
  - trains with run_training()
  - evaluates on the test split
  - compares against baselines
  - saves results to experiments/results/<experiment_name>/

Usage
-----
    # Run a single experiment
    python -m experiments.run_all --exp 1

    # Run all five (slow)
    python -m experiments.run_all --exp all

    # Programmatic
    from experiments.run_all import run_experiment_1, run_experiment_5
    run_experiment_1(n_specialist_episodes=50, n_orchestrator_episodes=100)
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import CFG
from environment.data_loader import DataLoader
from environment.market_env import MarketEnv
from experiments.baselines import evaluate_baselines
from orchestrator.portfolio import Portfolio, build_portfolio
from training.evaluate import (
    compare_strategies,
    evaluate_portfolio,
    plot_learning_curves,
    RESULTS_DIR,
)
from training.train import run_training

# Per-experiment episode budgets (small defaults; scale up for real runs)
_N_SPEC = 200      # specialist pre-training episodes per agent
_N_ORCH = 500      # orchestrator training episodes


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

def _make_test_env(cfg=CFG) -> MarketEnv:
    """Load data and return a MarketEnv backed by the test split."""
    loader = DataLoader(tickers=cfg.TICKERS, start=cfg.START_DATE, end=cfg.END_DATE)
    loader.load()
    test_features, test_prices = loader.get_split("test")
    return MarketEnv(features=test_features, close_prices=test_prices, seed=cfg.SEED)


def _baselines_for(test_env: MarketEnv, save_dir: Path, n_ep: int = 10) -> Dict:
    return evaluate_baselines(test_env, n_episodes=n_ep, save_dir=save_dir)


def _results_dir(name: str) -> Path:
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _print_header(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


# ──────────────────────────────────────────────────────────────────────
# Experiment 1 — Baseline DQN (single MacroAgent, no orchestration)
# ──────────────────────────────────────────────────────────────────────

def run_experiment_1(
    n_specialist_episodes: int = _N_SPEC,
    n_orchestrator_episodes: int = 0,     # orchestrator phase skipped
) -> Dict:
    """
    Baseline: one MacroAgent trained alone, no bandit or coordinator.

    We use run_training() with a single-agent portfolio (only MacroAgent)
    and set n_orchestrator_episodes=0 to skip Phase 2.
    """
    _print_header("Experiment 1 — Baseline DQN (MacroAgent only)")

    save_dir = _results_dir("exp1_baseline_dqn")
    cfg = dataclasses.replace(CFG, EXPERIMENT_NAME="exp1_baseline_dqn")

    # Load data, build env + single-agent portfolio
    loader = DataLoader(tickers=cfg.TICKERS, start=cfg.START_DATE, end=cfg.END_DATE)
    loader.load()
    train_features, train_prices = loader.get_split("train")
    train_env = MarketEnv(train_features, train_prices, seed=cfg.SEED)
    obs_dim = train_env.observation_space.shape[0]

    from agents.specialist import MacroAgent
    from orchestrator.bandit import ContextualUCBBandit
    from orchestrator.coordinator import RewardCoordinator
    from orchestrator.llm_regime import LLMRegimeDetector

    macro = MacroAgent(obs_dim=obs_dim, n_stocks=cfg.N_STOCKS)
    # Wrap in a trivial single-agent portfolio so we reuse the training API
    single_portfolio = Portfolio(
        agents=[macro],
        bandit=ContextualUCBBandit(1, [macro.name]),
        coordinator=RewardCoordinator([macro]),
        regime_detector=LLMRegimeDetector(),
    )

    result = run_training(
        cfg=cfg,
        n_specialist_episodes=n_specialist_episodes,
        n_orchestrator_episodes=n_orchestrator_episodes,
        experiment_name="exp1_baseline_dqn",
    )

    # Evaluate
    test_env = _make_test_env(cfg)
    trained_portfolio = result["portfolio"]
    baselines = _baselines_for(test_env, save_dir)
    strategies = {"alphaportfolio_dqn": trained_portfolio, **_wrap_baselines(baselines, test_env)}
    metrics = compare_strategies(strategies, test_env, n_episodes=10, save_dir=save_dir)

    plot_learning_curves(
        {"MacroAgent": result["specialist_stats"].get("macro_agent", [])},
        save_path=save_dir / "learning_curve.png",
    )

    print(f"\n[Exp 1] Results saved to {save_dir}")
    return {"metrics": metrics, "baselines": baselines, "result": result}


# ──────────────────────────────────────────────────────────────────────
# Experiment 2 — DQN + UCB bandit (no reward sharing)
# ──────────────────────────────────────────────────────────────────────

def run_experiment_2(
    n_specialist_episodes: int = _N_SPEC,
    n_orchestrator_episodes: int = _N_ORCH,
) -> Dict:
    """
    DQN specialists + UCB bandit routing.
    Coordinator alpha=1.0 so every agent gets its raw local reward (no sharing).
    """
    _print_header("Experiment 2 — DQN + UCB bandit (no reward sharing)")

    save_dir = _results_dir("exp2_dqn_ucb")
    # alpha=1.0: blended = local*1.0 + global*0.0 = no sharing
    cfg = dataclasses.replace(
        CFG, COORD_ALPHA=1.0, EXPERIMENT_NAME="exp2_dqn_ucb"
    )

    result = run_training(
        cfg=cfg,
        n_specialist_episodes=n_specialist_episodes,
        n_orchestrator_episodes=n_orchestrator_episodes,
        experiment_name="exp2_dqn_ucb",
    )

    test_env = _make_test_env(cfg)
    trained_portfolio = result["portfolio"]
    baselines = _baselines_for(test_env, save_dir)
    strategies = {"alphaportfolio_ucb": trained_portfolio, **_wrap_baselines(baselines, test_env)}
    metrics = compare_strategies(strategies, test_env, n_episodes=10, save_dir=save_dir)

    plot_learning_curves(
        _orch_curves(result),
        save_path=save_dir / "learning_curve.png",
    )

    print(f"\n[Exp 2] Results saved to {save_dir}")
    return {"metrics": metrics, "baselines": baselines, "result": result}


# ──────────────────────────────────────────────────────────────────────
# Experiment 3 — DQN + UCB + multi-agent reward sharing
# ──────────────────────────────────────────────────────────────────────

def run_experiment_3(
    n_specialist_episodes: int = _N_SPEC,
    n_orchestrator_episodes: int = _N_ORCH,
) -> Dict:
    """
    Full multi-agent setup: UCB bandit + coordinator reward sharing (alpha=0.7).
    LLM regime detection stays off.
    """
    _print_header("Experiment 3 — DQN + UCB + coordinator reward sharing")

    save_dir = _results_dir("exp3_dqn_ucb_coord")
    cfg = dataclasses.replace(
        CFG, COORD_ALPHA=0.7, EXPERIMENT_NAME="exp3_dqn_ucb_coord"
    )

    result = run_training(
        cfg=cfg,
        n_specialist_episodes=n_specialist_episodes,
        n_orchestrator_episodes=n_orchestrator_episodes,
        experiment_name="exp3_dqn_ucb_coord",
    )

    test_env = _make_test_env(cfg)
    trained_portfolio = result["portfolio"]
    baselines = _baselines_for(test_env, save_dir)
    strategies = {"alphaportfolio_coord": trained_portfolio, **_wrap_baselines(baselines, test_env)}
    metrics = compare_strategies(strategies, test_env, n_episodes=10, save_dir=save_dir)

    plot_learning_curves(
        _orch_curves(result),
        save_path=save_dir / "learning_curve.png",
    )

    print(f"\n[Exp 3] Results saved to {save_dir}")
    return {"metrics": metrics, "baselines": baselines, "result": result}


# ──────────────────────────────────────────────────────────────────────
# Experiment 4 — Full system with LLM regime detection
# ──────────────────────────────────────────────────────────────────────

def run_experiment_4(
    n_specialist_episodes: int = _N_SPEC,
    n_orchestrator_episodes: int = _N_ORCH,
) -> Dict:
    """
    Complete AlphaPortfolio: UCB + coordinator + LLM regime classification.

    Requires ANTHROPIC_API_KEY to be set; if absent the LLM detector
    silently falls back to the volatility heuristic.
    """
    _print_header("Experiment 4 — Full system (UCB + coord + LLM regime)")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "  WARNING: ANTHROPIC_API_KEY not set — LLM calls will fall back "
            "to the volatility heuristic. Set the key to enable real LLM regime detection."
        )

    save_dir = _results_dir("exp4_full_system")
    cfg = dataclasses.replace(
        CFG,
        USE_LLM_REGIME=True,
        COORD_ALPHA=0.7,
        EXPERIMENT_NAME="exp4_full_system",
    )

    result = run_training(
        cfg=cfg,
        n_specialist_episodes=n_specialist_episodes,
        n_orchestrator_episodes=n_orchestrator_episodes,
        experiment_name="exp4_full_system",
    )

    test_env = _make_test_env(cfg)
    trained_portfolio = result["portfolio"]
    baselines = _baselines_for(test_env, save_dir)
    strategies = {"alphaportfolio_full": trained_portfolio, **_wrap_baselines(baselines, test_env)}
    metrics = compare_strategies(strategies, test_env, n_episodes=10, save_dir=save_dir)

    # Log LLM detector stats
    llm_stats = trained_portfolio.regime_detector.get_stats()
    print(f"\n  LLM regime stats: {llm_stats}")

    plot_learning_curves(
        _orch_curves(result),
        save_path=save_dir / "learning_curve.png",
    )

    print(f"\n[Exp 4] Results saved to {save_dir}")
    return {"metrics": metrics, "baselines": baselines, "result": result, "llm_stats": llm_stats}


# ──────────────────────────────────────────────────────────────────────
# Experiment 5 — Ablation: COORD_ALPHA sweep
# ──────────────────────────────────────────────────────────────────────

def run_experiment_5(
    alphas: tuple = (0.0, 0.7, 1.0),
    n_specialist_episodes: int = _N_SPEC,
    n_orchestrator_episodes: int = _N_ORCH,
) -> Dict:
    """
    Ablation study: how does COORD_ALPHA affect performance?

    alpha=0.0 — pure global signal (fully cooperative team reward)
    alpha=0.7 — default mix
    alpha=1.0 — pure local signal (no coordination)

    Runs one full training + evaluation per alpha value and overlays
    learning curves for comparison.
    """
    _print_header("Experiment 5 — Ablation: COORD_ALPHA sweep")

    save_dir = _results_dir("exp5_ablation_alpha")
    all_results: Dict[str, Dict] = {}
    learning_curves: Dict[str, list] = {}

    for alpha in alphas:
        label = f"alpha={alpha:.1f}"
        print(f"\n  --- {label} ---")
        exp_name = f"exp5_alpha_{str(alpha).replace('.', '_')}"
        cfg = dataclasses.replace(
            CFG,
            COORD_ALPHA=alpha,
            EXPERIMENT_NAME=exp_name,
        )

        result = run_training(
            cfg=cfg,
            n_specialist_episodes=n_specialist_episodes,
            n_orchestrator_episodes=n_orchestrator_episodes,
            experiment_name=exp_name,
        )

        test_env = _make_test_env(cfg)
        trained_portfolio = result["portfolio"]
        metrics = evaluate_portfolio(trained_portfolio, test_env, n_episodes=10)

        all_results[label] = {
            "alpha": alpha,
            "metrics": metrics,
            "portfolio": trained_portfolio,
        }
        learning_curves[label] = result.get("orchestrator_stats", [])

        print(
            f"  {label}  ret={metrics['cumulative_return']:+.4f}  "
            f"sharpe={metrics['sharpe_ratio']:.3f}  "
            f"mdd={metrics['max_drawdown']:.4f}"
        )

    # Compare all alpha variants on a single plot
    plot_learning_curves(
        learning_curves,
        save_path=save_dir / "ablation_learning_curves.png",
    )

    # Metrics comparison across alpha values
    metrics_summary = {lbl: v["metrics"] for lbl, v in all_results.items()}
    from training.evaluate import plot_metrics_comparison, save_results
    plot_metrics_comparison(
        metrics_summary,
        save_path=save_dir / "ablation_metrics_comparison.png",
    )
    save_results(metrics_summary, save_dir, filename="ablation_metrics.json")

    # Also run baselines once for reference
    test_env_base = _make_test_env()
    baselines = _baselines_for(test_env_base, save_dir)

    print(f"\n[Exp 5] Ablation results saved to {save_dir}")
    return {"all_results": all_results, "baselines": baselines}


# ──────────────────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────────────────

def run_all(
    n_specialist_episodes: int = _N_SPEC,
    n_orchestrator_episodes: int = _N_ORCH,
) -> None:
    """Run all five experiments sequentially."""
    print("\nAlphaPortfolio — Full Experiment Suite")
    print(f"  specialist episodes : {n_specialist_episodes} per agent")
    print(f"  orchestrator episodes: {n_orchestrator_episodes}")
    print(f"  results root        : {RESULTS_DIR.resolve()}")

    kw = dict(
        n_specialist_episodes=n_specialist_episodes,
        n_orchestrator_episodes=n_orchestrator_episodes,
    )
    run_experiment_1(**kw)
    run_experiment_2(**kw)
    run_experiment_3(**kw)
    run_experiment_4(**kw)
    run_experiment_5(
        n_specialist_episodes=n_specialist_episodes,
        n_orchestrator_episodes=n_orchestrator_episodes,
    )
    print(f"\nAll experiments complete. Results in {RESULTS_DIR.resolve()}")


# ──────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────

def _orch_curves(result: Dict) -> Dict[str, list]:
    """Extract per-agent Phase-1 + orchestrator curves from a run_training result."""
    curves: Dict[str, list] = {}
    for name, stats in result.get("specialist_stats", {}).items():
        curves[f"{name} (phase1)"] = stats
    orch = result.get("orchestrator_stats")
    if orch:
        curves["orchestrator (phase2)"] = orch
    return curves


def _wrap_baselines(baselines_metrics: Dict, env: MarketEnv) -> Dict:
    """
    Return baseline agents keyed by name so compare_strategies() can
    re-run them on the same env for fair trajectory comparison.
    """
    from experiments.baselines import (
        BuyAndHoldAgent,
        EqualWeightRebalanceAgent,
        RandomAgent,
    )
    return {
        "random":                  RandomAgent(n_stocks=env.n_stocks),
        "buy_and_hold":            BuyAndHoldAgent(n_stocks=env.n_stocks),
        "equal_weight_rebalance":  EqualWeightRebalanceAgent(n_stocks=env.n_stocks),
    }


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaPortfolio experiment runner")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "all"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument("--spec-ep",  type=int, default=_N_SPEC,  help="Specialist episodes per agent")
    parser.add_argument("--orch-ep",  type=int, default=_N_ORCH,  help="Orchestrator episodes")
    args = parser.parse_args()

    kw = dict(
        n_specialist_episodes=args.spec_ep,
        n_orchestrator_episodes=args.orch_ep,
    )

    dispatch = {
        "1": run_experiment_1,
        "2": run_experiment_2,
        "3": run_experiment_3,
        "4": run_experiment_4,
        "5": lambda **k: run_experiment_5(**k),
    }

    if args.exp == "all":
        run_all(**kw)
    else:
        dispatch[args.exp](**kw)
