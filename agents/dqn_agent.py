"""
dqn_agent.py
------------
Deep Q-Network agent for multi-stock portfolio management.

Architecture
------------
  QNetwork: 3-layer MLP
    obs_dim  →  hidden_dim (ReLU)
             →  hidden_dim (ReLU)
             →  n_stocks * n_actions

  DQNAgent manages one online_net + one target_net.
  The joint output is reshaped to (n_stocks, n_actions) for per-stock
  action selection and TD-target computation.

Variants
--------
  Standard DQN:  target = r + γ^n · max_a Q_tgt(s', a)
  Double DQN:    target = r + γ^n · Q_tgt(s', argmax_a Q_online(s', a))
                 Controlled by USE_DOUBLE_DQN (reads from CFG if present,
                 defaults to True).

Epsilon decay
-------------
  Linear annealing from EPSILON_START to EPSILON_END over EPSILON_DECAY
  steps, decremented in store_transition() (training only).
  During evaluation call agent.eval_mode() to force greedy selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import CFG
from agents.base_agent import BaseAgent
from agents.replay_buffer import NStepBuffer, ReplayBuffer


# ──────────────────────────────────────────────────────────────────────
# Q-Network
# ──────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    3-layer MLP Q-function.

    Parameters
    ----------
    obs_dim    : dimension of the flattened observation vector
    out_dim    : n_stocks * n_actions  (e.g. 5 * 3 = 15)
    hidden_dim : width of each hidden layer (default: CFG.HIDDEN_DIM or 256)
    """

    def __init__(self, obs_dim: int, out_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        # Smaller output layer init for stable early Q-values
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────

class DQNAgent(BaseAgent):
    """
    DQN agent for the AlphaPortfolio MarketEnv.

    Parameters
    ----------
    obs_dim      : observation space dimension (from env.observation_space.shape[0])
    n_stocks     : number of tradeable assets
    n_actions    : discrete actions per stock (default 3: sell/hold/buy)
    cfg          : config object; defaults to the global CFG singleton
    name         : agent identifier string
    """

    def __init__(
        self,
        obs_dim: int,
        n_stocks: int,
        n_actions: int = 3,
        cfg=CFG,
        name: str = "dqn_agent",
    ) -> None:
        super().__init__(name)

        self.obs_dim = obs_dim
        self.n_stocks = n_stocks
        self.n_actions = n_actions
        self.cfg = cfg

        # --- Hyperparameters (forward-compatible with future config additions) ---
        self._hidden_dim: int = getattr(cfg, "HIDDEN_DIM", 256)
        self._use_double: bool = getattr(cfg, "USE_DOUBLE_DQN", True)
        self._grad_clip: float = getattr(cfg, "GRAD_CLIP", 10.0)
        self._n_step: int = getattr(cfg, "N_STEP_RETURN", 1)

        # --- Device ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Networks ---
        out_dim = n_stocks * n_actions
        self.online_net = QNetwork(obs_dim, out_dim, self._hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, out_dim, self._hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net is never in training mode

        # --- Optimizer ---
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=cfg.LEARNING_RATE
        )

        # --- Replay buffer (wrapped in NStepBuffer if n_step > 1) ---
        raw_buffer = ReplayBuffer(cfg.REPLAY_BUFFER_SIZE)
        if self._n_step > 1:
            self._replay: Union[NStepBuffer, ReplayBuffer] = NStepBuffer(
                self._n_step, cfg.GAMMA, raw_buffer
            )
        else:
            self._replay = raw_buffer

        # --- Exploration state ---
        self._epsilon: float = cfg.EPSILON_START
        self._eval: bool = False       # True → always greedy

        # --- Training counters ---
        self._update_steps: int = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return 0.0 if self._eval else self._epsilon

    def eval_mode(self) -> None:
        self._eval = True
        self.online_net.eval()

    def train_mode(self) -> None:
        self._eval = False
        self.online_net.train()

    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Epsilon-greedy action selection.

        Returns
        -------
        action : int array of shape (n_stocks,), values in {0, 1, 2}
        """
        if not self._eval and np.random.random() < self._epsilon:
            return np.array(
                [np.random.randint(self.n_actions) for _ in range(self.n_stocks)],
                dtype=np.int64,
            )

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(obs_t)               # (1, n_stocks * n_actions)
        q = q.view(self.n_stocks, self.n_actions)    # (n_stocks, n_actions)
        return q.argmax(dim=-1).cpu().numpy().astype(np.int64)

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._total_steps += 1
        self._replay.push(obs, action, reward, next_obs, done)
        self._decay_epsilon()

    def update(self) -> Optional[float]:
        """
        One gradient step of DQN.

        Returns
        -------
        loss : TD loss (float), or None if buffer has fewer samples than BATCH_SIZE.
        """
        if not self._replay.is_ready(self.cfg.BATCH_SIZE):
            return None

        batch = self._replay.sample(self.cfg.BATCH_SIZE)
        obs      = batch.obs.to(self.device)       # (B, obs_dim)
        action   = batch.action.to(self.device)    # (B, n_stocks)
        reward   = batch.reward.to(self.device)    # (B,)
        next_obs = batch.next_obs.to(self.device)  # (B, obs_dim)
        done     = batch.done.to(self.device)      # (B,)

        B = obs.shape[0]

        # --- Current Q(s, a) for each stock ---
        q_all = self.online_net(obs).view(B, self.n_stocks, self.n_actions)
        # gather Q-values for the actions that were actually taken
        q_taken = q_all.gather(2, action.unsqueeze(2)).squeeze(2)  # (B, n_stocks)

        # --- Target Q-values ---
        with torch.no_grad():
            if self._use_double:
                # Double DQN: online net selects, target net evaluates
                next_actions = (
                    self.online_net(next_obs)
                    .view(B, self.n_stocks, self.n_actions)
                    .argmax(dim=2, keepdim=True)           # (B, n_stocks, 1)
                )
                next_q = (
                    self.target_net(next_obs)
                    .view(B, self.n_stocks, self.n_actions)
                )
                next_max_q = next_q.gather(2, next_actions).squeeze(2)  # (B, n_stocks)
            else:
                next_max_q = (
                    self.target_net(next_obs)
                    .view(B, self.n_stocks, self.n_actions)
                    .max(dim=2).values                     # (B, n_stocks)
                )

            # Shared reward broadcast across all stocks
            r = reward.unsqueeze(1).expand(-1, self.n_stocks)      # (B, n_stocks)
            mask = (1.0 - done).unsqueeze(1).expand(-1, self.n_stocks)

            targets = r + mask * (self.cfg.GAMMA ** self._n_step) * next_max_q

        loss = F.mse_loss(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self._grad_clip)
        self.optimizer.step()

        self._update_steps += 1

        # Hard target network update
        if self._update_steps % self.cfg.TARGET_UPDATE_FREQ == 0:
            self.hard_update_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def hard_update_target(self) -> None:
        """Copy online → target weights exactly."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update_target(self, tau: Optional[float] = None) -> None:
        """
        Polyak averaging: θ_tgt ← τ·θ_online + (1−τ)·θ_tgt
        tau defaults to CFG.TAU.
        """
        tau = tau if tau is not None else self.cfg.TAU
        for p_tgt, p_online in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p_online.data)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self._epsilon,
                "total_steps": self._total_steps,
                "update_steps": self._update_steps,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._epsilon = ckpt["epsilon"]
        self._total_steps = ckpt["total_steps"]
        self._update_steps = ckpt["update_steps"]

    # ------------------------------------------------------------------
    # Stats for logging
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update(
            {
                "epsilon": self._epsilon,
                "update_steps": self._update_steps,
                "buffer_size": len(self._replay),
                "double_dqn": self._use_double,
            }
        )
        return stats

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _decay_epsilon(self) -> None:
        """Linear epsilon annealing from EPSILON_START to EPSILON_END."""
        progress = self._total_steps / max(self.cfg.EPSILON_DECAY, 1)
        self._epsilon = max(
            self.cfg.EPSILON_END,
            self.cfg.EPSILON_START
            - progress * (self.cfg.EPSILON_START - self.cfg.EPSILON_END),
        )
