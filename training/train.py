"""
train.py
--------
Two-phase training loop for AlphaPortfolio.

Phase 1 — Specialist pre-training
    Each of the three specialist agents (Momentum, MeanReversion, Macro)
    is trained independently on the full training split for
    n_specialist_episodes episodes.  This gives each agent a reasonable
    policy before the orchestrator begins routing.

Phase 2 — Orchestrator training
    The full Portfolio (bandit + coordinator + all three agents) is trained
    on the training split for n_orchestrator_episodes episodes.  Agents
    continue learning via coordinator-blended rewards; the bandit learns
    which specialist to select per volatility context.

Entry point
-----------
    from training.train import run_training
    run_training()               # uses config defaults
    run_training(cfg, name="exp2", n_specialist_episodes=100)
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):          # minimal fallback
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", "?")
        print(f"{desc} ({total} episodes)")
        return it

from config import CFG
from environment.data_loader import DataLoader
from environment.market_env import MarketEnv
from orchestrator.portfolio import Portfolio, build_portfolio
from utils.logger import EpisodeTracker, ExperimentLogger

# How many recent price rows to pass to the orchestrator for regime detection
_REGIME_WINDOW = 20


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _recent_prices(env: MarketEnv, window: int = _REGIME_WINDOW) -> np.ndarray:
    """Extract the last `window` rows of closing prices at current env timestep."""
    t = env._t
    start = max(0, t - window)
    return env.close_prices[start : t + 1]


def _agent_selection_pct(freq: Dict[str, int]) -> Dict[str, float]:
    total = sum(freq.values()) + 1e-8
    return {k: v / total for k, v in freq.items()}


# ──────────────────────────────────────────────────────────────────────
# Phase 1: sequential specialist pre-training
# ──────────────────────────────────────────────────────────────────────

def train_specialists(
    portfolio: Portfolio,
    env: MarketEnv,
    logger: ExperimentLogger,
    n_episodes: int,
    checkpoint_freq: int = 50,
) -> Dict[str, List[dict]]:
    """
    Train each specialist agent independently on the training split.

    Returns
    -------
    {agent_name: [episode_stats_dict, ...]}
    """
    all_stats: Dict[str, List[dict]] = {}

    for agent in portfolio.agents:
        print(f"\n{'='*60}")
        print(f"Phase 1 | Training {agent.name}  ({n_episodes} episodes)")
        print(f"{'='*60}")

        tracker = EpisodeTracker()
        agent_stats: List[dict] = []

        pbar = tqdm(range(n_episodes), desc=f"  {agent.name}", unit="ep")
        for ep in pbar:
            tracker.start_episode()
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            losses: List[float] = []

            while not done:
                action = agent.select_action(obs)
                next_obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                agent.store_transition(obs, action, reward, next_obs, done)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                tracker.log_step(reward)
                obs = next_obs
                total_reward += reward

            stats = tracker.end_episode(total_reward, env.episode_length)
            stats["mean_loss"] = float(np.mean(losses)) if losses else 0.0
            stats["agent"] = agent.name
            logger.log(stats, step=tracker.episode)
            agent_stats.append(stats)

            # Checkpoint
            if (ep + 1) % checkpoint_freq == 0:
                path = logger.checkpoint_dir / f"{agent.name}_phase1_ep{ep+1:06d}.pt"
                agent.save(path)

            pbar.set_postfix(
                ret=f"{stats['episode_return']:+.5f}",
                eps=f"{agent.epsilon:.3f}",
                loss=f"{stats['mean_loss']:.5f}",
                m100=f"{stats['mean_return_100']:+.5f}",
            )

        all_stats[agent.name] = agent_stats

    return all_stats


# ──────────────────────────────────────────────────────────────────────
# Phase 2: orchestrator training
# ──────────────────────────────────────────────────────────────────────

def train_orchestrator(
    portfolio: Portfolio,
    env: MarketEnv,
    logger: ExperimentLogger,
    n_episodes: int,
    checkpoint_freq: int = 50,
) -> List[dict]:
    """
    Train the full Portfolio (bandit + coordinator + all agents).

    Returns
    -------
    list of episode stats dicts
    """
    print(f"\n{'='*60}")
    print(f"Phase 2 | Orchestrator training  ({n_episodes} episodes)")
    print(f"{'='*60}\n")

    tracker = EpisodeTracker()
    episode_stats: List[dict] = []
    cum_selection: Dict[str, int] = {a.name: 0 for a in portfolio.agents}

    pbar = tqdm(range(n_episodes), desc="  Orchestrator", unit="ep")
    for ep in pbar:
        tracker.start_episode()
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step_losses: Dict[str, List[float]] = {a.name: [] for a in portfolio.agents}
        ep_selection: Dict[str, int] = {a.name: 0 for a in portfolio.agents}

        while not done:
            recent_px = _recent_prices(env)
            action, meta = portfolio.select_action(obs, recent_px)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            losses = portfolio.update(obs, action, reward, next_obs, done)
            for name, loss in losses.items():
                if loss is not None:
                    step_losses[name].append(loss)

            ep_selection[meta["agent"]] += 1
            cum_selection[meta["agent"]] += 1
            tracker.log_step(reward)
            obs = next_obs
            total_reward += reward

        stats = tracker.end_episode(total_reward, env.episode_length)

        # Per-agent mean loss this episode
        for name, ls in step_losses.items():
            stats[f"loss_{name}"] = float(np.mean(ls)) if ls else 0.0

        # Agent selection frequency (episode-level and cumulative)
        sel_pct = _agent_selection_pct(ep_selection)
        for name, pct in sel_pct.items():
            stats[f"sel_{name}"] = pct

        stats["regime"] = meta.get("regime", "unknown")
        stats["context"] = meta.get("context", "med")

        logger.log(stats, step=tracker.episode)
        episode_stats.append(stats)

        # Save checkpoints
        if (ep + 1) % checkpoint_freq == 0:
            for agent in portfolio.agents:
                path = logger.checkpoint_dir / f"{agent.name}_phase2_ep{ep+1:06d}.pt"
                agent.save(path)

        top_agent = max(cum_selection, key=cum_selection.get)
        pbar.set_postfix(
            ret=f"{stats['episode_return']:+.5f}",
            m100=f"{stats['mean_return_100']:+.5f}",
            top=top_agent.replace("_agent", ""),
            regime=meta.get("regime", "?")[:4],
        )

    return episode_stats


# ──────────────────────────────────────────────────────────────────────
# Phase 1.5: warm-up in orchestrated environment
# ──────────────────────────────────────────────────────────────────────

def train_warmup(portfolio, env, logger, n_episodes: int) -> None:
    if n_episodes == 0:
        return
    print("\n" + "="*60)
    print(f"Phase 1.5 | Warm-up ({n_episodes} eps per agent in orchestrated env)")
    print("="*60 + "\n")
    for agent in portfolio.agents:
        pbar = tqdm(range(n_episodes), desc=f"  warmup {agent.name}", unit="ep")
        for ep in pbar:
            obs, _ = env.reset()
            done = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                agent.store_transition(obs, action, reward, next_obs, done)
                agent.update()
                obs = next_obs
        print(f"  {agent.name} warmup complete")


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def run_training(
    cfg=CFG,
    n_specialist_episodes: int = 200,
    n_orchestrator_episodes: int = 500,
    n_warmup_episodes: int = 0,
    checkpoint_freq: int = 50,
    experiment_name: Optional[str] = None,
) -> Dict:
    """
    Full two-phase training run.

    Parameters
    ----------
    cfg                      : Config object (use dataclasses.replace(CFG, ...) for variants)
    n_specialist_episodes    : episodes per agent in Phase 1
    n_orchestrator_episodes  : episodes for orchestrator in Phase 2
    checkpoint_freq          : save checkpoint every N episodes
    experiment_name          : override for the run directory name

    Returns
    -------
    dict with keys 'specialist_stats', 'orchestrator_stats', 'logger', 'portfolio'
    """
    t0 = time.time()

    # --- Data ---
    print("[Train] Loading market data...")
    loader = DataLoader(
        tickers=cfg.TICKERS,
        start=cfg.START_DATE,
        end=cfg.END_DATE,
    )
    loader.load()
    train_features, train_prices = loader.get_split("train")

    # --- Environment ---
    train_env = MarketEnv(
        features=train_features,
        close_prices=train_prices,
        seed=cfg.SEED,
    )
    obs_dim = train_env.observation_space.shape[0]

    # --- Portfolio ---
    portfolio = build_portfolio(obs_dim=obs_dim, n_stocks=cfg.N_STOCKS)

    # --- Logger ---
    logger = ExperimentLogger(
        log_dir=cfg.LOG_DIR,
        experiment_name=experiment_name or cfg.EXPERIMENT_NAME,
    )
    logger.log_hyperparams(cfg)

    # --- Phase 1 ---
    specialist_stats = train_specialists(
        portfolio=portfolio,
        env=train_env,
        logger=logger,
        n_episodes=n_specialist_episodes,
        checkpoint_freq=checkpoint_freq,
    )

    # --- Phase 1.5 ---
    train_warmup(
        portfolio=portfolio,
        env=train_env,
        logger=logger,
        n_episodes=n_warmup_episodes,
    )

    # --- Phase 2 ---
    orchestrator_stats = train_orchestrator(
        portfolio=portfolio,
        env=train_env,
        logger=logger,
        n_episodes=n_orchestrator_episodes,
        checkpoint_freq=checkpoint_freq,
    )

    elapsed = time.time() - t0
    print(f"\n[Train] Complete in {elapsed/60:.1f} min  |  run dir: {logger.run_dir}")

    return {
        "specialist_stats": specialist_stats,
        "orchestrator_stats": orchestrator_stats,
        "logger": logger,
        "portfolio": portfolio,
        "loader": loader,
    }


if __name__ == "__main__":
    run_training()
