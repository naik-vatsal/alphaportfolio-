"""
portfolio.py
------------
Top-level orchestrator that combines all AlphaPortfolio components into
a single, clean interface for the training loop.

Responsibilities
----------------
1. Regime detection  — asks LLMRegimeDetector for current market regime
2. Context mapping   — maps recent price volatility to "low"/"med"/"high"
3. Agent selection   — delegates to ContextualUCBBandit
4. Reward sharing    — delegates to RewardCoordinator
5. History tracking  — records (step, agent, regime, context, reward) tuples

select_action(obs, recent_prices) → action, metadata
update(obs, action, env_reward, next_obs, done) → {agent: loss}

The training loop only needs to call these two methods.

Factory function
----------------
build_portfolio(obs_dim, n_stocks) instantiates all components from
config defaults and returns a ready-to-use Portfolio.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import CFG
from agents.specialist import MacroAgent, MeanReversionAgent, MomentumAgent
from orchestrator.bandit import ContextualUCBBandit
from orchestrator.coordinator import RewardCoordinator
from orchestrator.llm_regime import LLMRegimeDetector


# Volatility thresholds for the "low / med / high" context
_VOL_LOW_THRESHOLD: float = 0.010   # annualised daily std < 1 %
_VOL_HIGH_THRESHOLD: float = 0.020  # annualised daily std > 2 %

# How many recent days to use for regime / volatility estimation
_CONTEXT_WINDOW: int = 20


class Portfolio:
    """
    Combines bandit selection, coordinator reward sharing, and regime
    detection into a single training-loop-facing interface.

    Parameters
    ----------
    agents          : list of specialist agents (order defines bandit arm indices)
    bandit          : ContextualUCBBandit
    coordinator     : RewardCoordinator
    regime_detector : LLMRegimeDetector
    """

    def __init__(
        self,
        agents: List,
        bandit: ContextualUCBBandit,
        coordinator: RewardCoordinator,
        regime_detector: LLMRegimeDetector,
    ) -> None:
        self.agents = agents
        self.bandit = bandit
        self.coordinator = coordinator
        self.regime_detector = regime_detector

        self._n_agents = len(agents)
        self._agent_names: List[str] = [a.name for a in agents]

        # Step-level state (set in select_action, consumed in update)
        self._last_agent_idx: int = 0
        self._last_context: str = "med"
        self._last_regime: str = "uncertain"

        # Full selection history for analysis and reporting
        self._history: List[Dict[str, Any]] = []
        self._step: int = 0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        recent_prices: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Choose an action for the current environment step.

        Parameters
        ----------
        obs           : current observation, shape (obs_dim,)
        recent_prices : recent closing prices, shape (T, n_stocks)
                        Used for regime detection and volatility context.
                        Typically the last _CONTEXT_WINDOW days.

        Returns
        -------
        action   : int array of shape (n_stocks,), values in {0, 1, 2}
        metadata : dict with keys agent, regime, context, ucb_scores
        """
        # 1. Detect market regime (LLM or heuristic)
        regime = self.regime_detector.detect(
            np.empty((0,)),   # features not required by heuristic path
            recent_prices,
        )

        # 2. Map price volatility → bandit context
        context = _volatility_context(recent_prices)

        # 3. Bandit selects agent
        agent_idx = self.bandit.select(context)
        agent = self.agents[agent_idx]

        # 4. Selected agent proposes action
        action = agent.select_action(obs)

        # 5. Stash for update()
        self._last_agent_idx = agent_idx
        self._last_context = context
        self._last_regime = regime

        metadata = {
            "agent": agent.name,
            "agent_idx": agent_idx,
            "regime": regime,
            "context": context,
            "ucb_scores": self.bandit.ucb_scores(context),
            "step": self._step,
        }
        return action, metadata

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        env_reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Dict[str, Optional[float]]:
        """
        Store transitions, blend rewards, update all networks and bandit.

        All agents learn from every step (cooperative team setting).
        Each agent's stored reward is the coordinator-blended version of
        env_reward, which incorporates peer Q-value confidence signals.

        Parameters
        ----------
        obs        : observation before the step
        action     : action actually executed in the environment
        env_reward : scalar reward returned by the environment
        next_obs   : observation after the step
        done       : whether the episode terminated

        Returns
        -------
        {agent_name: loss_float_or_None}
        """
        # All agents share the same local reward (cooperative)
        local_rewards = {name: env_reward for name in self._agent_names}

        # Coordinator blends with confidence-weighted global reward
        blended = self.coordinator.compute_shared_rewards(obs, local_rewards)

        # All agents store their blended transition and update
        losses: Dict[str, Optional[float]] = {}
        for agent in self.agents:
            agent.store_transition(
                obs, action, blended[agent.name], next_obs, done
            )
            losses[agent.name] = agent.update()

        # Update bandit with selected agent's blended reward
        self.bandit.update(
            self._last_agent_idx,
            blended[self._agent_names[self._last_agent_idx]],
            self._last_context,
        )

        # Log step
        self._history.append(
            {
                "step": self._step,
                "agent": self._agent_names[self._last_agent_idx],
                "regime": self._last_regime,
                "context": self._last_context,
                "env_reward": env_reward,
                "blended_reward": blended[self._agent_names[self._last_agent_idx]],
            }
        )
        self._step += 1

        return losses

    # ------------------------------------------------------------------
    # Eval / train mode forwarding
    # ------------------------------------------------------------------

    def eval_mode(self) -> None:
        for agent in self.agents:
            agent.eval_mode()

    def train_mode(self) -> None:
        for agent in self.agents:
            agent.train_mode()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "portfolio_step": self._step,
            "bandit": self.bandit.get_stats(),
            "coordinator": self.coordinator.get_stats(),
            "regime_detector": self.regime_detector.get_stats(),
        }
        for agent in self.agents:
            stats[f"agent_{agent.name}"] = agent.get_stats()
        return stats

    def selection_summary(self) -> Dict[str, int]:
        """Count how many times each agent was selected."""
        counts: Dict[str, int] = {name: 0 for name in self._agent_names}
        for record in self._history:
            counts[record["agent"]] += 1
        return counts

    def regime_summary(self) -> Dict[str, int]:
        """Count how many steps were spent in each detected regime."""
        counts: Dict[str, int] = {}
        for record in self._history:
            r = record["regime"]
            counts[r] = counts.get(r, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"Portfolio(agents={self._agent_names}, "
            f"step={self._step})"
        )


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def build_portfolio(obs_dim: int, n_stocks: int) -> Portfolio:
    """
    Instantiate all components from config defaults.

    Parameters
    ----------
    obs_dim  : env.observation_space.shape[0]
    n_stocks : number of tradeable assets (CFG.N_STOCKS)

    Returns
    -------
    Ready-to-use Portfolio.
    """
    agents = [
        MomentumAgent(obs_dim=obs_dim, n_stocks=n_stocks),
        MeanReversionAgent(obs_dim=obs_dim, n_stocks=n_stocks),
        MacroAgent(obs_dim=obs_dim, n_stocks=n_stocks),
    ]

    bandit = ContextualUCBBandit(
        n_agents=len(agents),
        agent_names=[a.name for a in agents],
        c=getattr(CFG, "UCB_C", 2.0),
    )

    coordinator = RewardCoordinator(agents)

    regime_detector = LLMRegimeDetector(
        model=CFG.LLM_MODEL,
    )

    return Portfolio(
        agents=agents,
        bandit=bandit,
        coordinator=coordinator,
        regime_detector=regime_detector,
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _volatility_context(prices: np.ndarray) -> str:
    """
    Map recent price volatility to a bandit context string.

    Uses the std of daily returns over the last _CONTEXT_WINDOW days.
    Thresholds (configurable via module-level constants):
        low  : std < 1.0 %
        med  : 1.0 % <= std <= 2.0 %
        high : std > 2.0 %
    """
    if len(prices) < 2:
        return "med"

    window = prices[-min(_CONTEXT_WINDOW, len(prices)):]
    daily_ret = np.diff(window, axis=0) / (window[:-1] + 1e-8)
    vol = float(np.std(daily_ret))

    if vol < _VOL_LOW_THRESHOLD:
        return "low"
    if vol > _VOL_HIGH_THRESHOLD:
        return "high"
    return "med"
