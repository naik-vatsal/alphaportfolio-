"""
replay_buffer.py
----------------
Experience replay for AlphaPortfolio DQN agents.

Classes
-------
Transition      — namedtuple used throughout agents
ReplayBuffer    — circular buffer with uniform random sampling
NStepBuffer     — accumulates n transitions, computes discounted return,
                  then forwards the compressed transition to a ReplayBuffer
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch


# Shared transition type used by both buffer classes and the DQN update
Transition = namedtuple(
    "Transition", ["obs", "action", "reward", "next_obs", "done"]
)


# ──────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular uniform-sampling experience replay buffer.

    push() stores raw numpy arrays.
    sample() returns a Transition of stacked float32 tensors, ready for
    direct use in a PyTorch loss computation.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buf: list = [None] * capacity
        self._pos: int = 0
        self._size: int = 0

    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf[self._pos] = (
            obs.astype(np.float32, copy=False),
            np.asarray(action, dtype=np.int64),
            float(reward),
            next_obs.astype(np.float32, copy=False),
            bool(done),
        )
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        """
        Sample *batch_size* transitions uniformly.

        Returns
        -------
        Transition with tensor fields:
          obs       float32  (B, obs_dim)
          action    int64    (B, n_stocks)
          reward    float32  (B,)
          next_obs  float32  (B, obs_dim)
          done      float32  (B,)   — cast to float for loss masking
        """
        indices = random.sample(range(self._size), batch_size)
        obs, act, rew, nobs, done = zip(*[self._buf[i] for i in indices])
        return Transition(
            obs=torch.from_numpy(np.stack(obs)),
            action=torch.from_numpy(np.stack(act)),
            reward=torch.tensor(rew, dtype=torch.float32),
            next_obs=torch.from_numpy(np.stack(nobs)),
            done=torch.tensor(done, dtype=torch.float32),
        )

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, size={self._size})"


# ──────────────────────────────────────────────────────────────────────
# N-step wrapper
# ──────────────────────────────────────────────────────────────────────

class NStepBuffer:
    """
    Wraps a ReplayBuffer to store n-step discounted returns.

    For each transition (s_t, a_t), the compressed tuple pushed to the
    underlying buffer is:

        (s_t, a_t, G_{t:t+n}, s_{t+n}, done_{t+n})

    where  G_{t:t+n} = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1}

    At episode boundaries, remaining transitions are flushed with their
    available (< n) step returns so no experience is lost.

    When n == 1 this is equivalent to passing directly to ReplayBuffer.
    """

    def __init__(
        self, n: int, gamma: float, main_buffer: ReplayBuffer
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.gamma = gamma
        self.main = main_buffer
        # Each element: (obs, action, reward, next_obs, done)
        self._pending: deque = deque()

    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._pending.append(
            (obs.copy(), np.asarray(action, dtype=np.int64),
             float(reward), next_obs.copy(), bool(done))
        )

        # Commit oldest transition when we have a full n-step window
        if len(self._pending) >= self.n:
            self._commit_oldest()

        # On episode end, flush whatever is left in the window
        if done:
            while self._pending:
                self._commit_oldest()

    def _commit_oldest(self) -> None:
        """
        Compute discounted return for the oldest pending transition and
        push the result to the main ReplayBuffer.
        """
        G = 0.0
        terminal = False
        bootstrap_next_obs = self._pending[-1][3]  # default: furthest next obs

        for i, (_, _, r, nobs, d) in enumerate(self._pending):
            G += (self.gamma ** i) * r
            if d:
                terminal = True
                bootstrap_next_obs = nobs  # no bootstrapping past terminal
                break

        obs_0, action_0, _, _, _ = self._pending[0]
        self.main.push(obs_0, action_0, G, bootstrap_next_obs, terminal)
        self._pending.popleft()

    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Transition:
        """Delegate sampling to the underlying ReplayBuffer."""
        return self.main.sample(batch_size)

    def is_ready(self, batch_size: int) -> bool:
        return self.main.is_ready(batch_size)

    def __len__(self) -> int:
        return len(self.main)

    def __repr__(self) -> str:
        return (
            f"NStepBuffer(n={self.n}, gamma={self.gamma}, "
            f"pending={len(self._pending)}, main={self.main})"
        )
