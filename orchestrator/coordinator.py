"""
coordinator.py
--------------
Multi-agent reward sharing and Q-value confidence communication.

Reward sharing
--------------
Each specialist agent receives a *blended* reward rather than the raw
environment reward:

    blended_i = local_i * alpha + global * (1 - alpha)

    local_i  : environment reward attributed to agent i
    global   : confidence-weighted average of all local rewards
    alpha    : mixing coefficient (default 0.7; set COORD_ALPHA in config)

The global signal is weighted by each agent's Q-value confidence
(max Q-value of the online network at the current observation) so that
agents with stronger beliefs about the current state contribute more to
the shared signal.

Confidence communication
------------------------
get_confidences(obs) returns {agent_name: max_Q} for each agent that
exposes an `online_net` attribute.  The orchestrator / portfolio can
use these for logging, dynamic alpha adjustment, or routing decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config import CFG


class RewardCoordinator:
    """
    Blends local and global rewards across a team of agents.

    Parameters
    ----------
    agents : list of agent objects (must implement BaseAgent interface)
    alpha  : local-vs-global mixing weight (reads COORD_ALPHA from config,
             falls back to 0.7)
    """

    def __init__(self, agents: List, alpha: Optional[float] = None) -> None:
        self.agents = agents
        self.alpha: float = (
            alpha
            if alpha is not None
            else getattr(CFG, "COORD_ALPHA", 0.7)
        )
        self._agent_names: List[str] = [a.name for a in agents]

        # Running stats for diagnostics
        self._step: int = 0
        self._reward_history: Dict[str, List[float]] = {
            a.name: [] for a in agents
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_shared_rewards(
        self,
        obs: np.ndarray,
        local_rewards: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Blend local and confidence-weighted global rewards.

        Parameters
        ----------
        obs           : current observation (float32 array, shape obs_dim)
        local_rewards : {agent_name: env_reward} for each agent

        Returns
        -------
        blended : {agent_name: blended_reward}
        """
        # --- Confidence-weighted global reward ---
        confidences = self.get_confidences(obs)
        total_conf = sum(confidences.values()) + 1e-8

        global_reward = sum(
            (confidences[name] / total_conf) * local_rewards[name]
            for name in self._agent_names
        )

        # --- Blend ---
        blended: Dict[str, float] = {}
        for name in self._agent_names:
            local = local_rewards[name]
            blended[name] = local * self.alpha + global_reward * (1.0 - self.alpha)
            self._reward_history[name].append(blended[name])

        self._step += 1
        return blended

    def get_confidences(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Query each agent's online Q-network for its maximum Q-value.

        The max Q-value is a proxy for the agent's confidence: a high
        value means the agent believes it knows a good action in this
        state.  Agents without an `online_net` attribute receive a
        neutral confidence of 1.0.

        Parameters
        ----------
        obs : float32 array of shape (obs_dim,)

        Returns
        -------
        {agent_name: confidence_float}
        """
        confidences: Dict[str, float] = {}
        for agent in self.agents:
            confidences[agent.name] = _query_q_confidence(agent, obs)
        return confidences

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "alpha": self.alpha,
            "coord_step": self._step,
        }
        for name, hist in self._reward_history.items():
            if hist:
                stats[f"{name}_mean_blended"] = float(np.mean(hist[-100:]))
        return stats

    def mean_blended_rewards(self, window: int = 100) -> Dict[str, float]:
        return {
            name: float(np.mean(hist[-window:])) if hist else 0.0
            for name, hist in self._reward_history.items()
        }

    def __repr__(self) -> str:
        return (
            f"RewardCoordinator("
            f"agents={self._agent_names}, alpha={self.alpha}, "
            f"step={self._step})"
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _query_q_confidence(agent, obs: np.ndarray) -> float:
    """
    Return max Q-value of agent's online network at obs.
    Falls back to 1.0 if the agent has no online network.
    """
    if not (hasattr(agent, "online_net") and hasattr(agent, "device")):
        return 1.0

    obs_t = (
        torch.from_numpy(np.asarray(obs, dtype=np.float32))
        .unsqueeze(0)
        .to(agent.device)
    )
    with torch.no_grad():
        q_vals = agent.online_net(obs_t)
    return float(q_vals.max().item())
