"""
base_agent.py
-------------
Abstract base class for all AlphaPortfolio agents.

Every concrete agent (DQN, PPO, specialist wrappers) must implement:
  select_action  — obs → action array
  store_transition — push (s, a, r, s', done) to internal buffer
  update         — one gradient step; returns loss or None

Concrete helpers (save / load / get_stats) are provided here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


class BaseAgent(ABC):
    """Abstract base for all AlphaPortfolio agents."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def epsilon(self) -> float:
        """Current exploration rate. Override in subclass if applicable."""
        return 0.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Choose an action for the given observation.

        Parameters
        ----------
        obs : float32 array of shape (obs_dim,)

        Returns
        -------
        action : int array of shape (n_stocks,), values in {0, 1, 2}
        """

    @abstractmethod
    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store one environment transition. Also responsible for:
          - incrementing total_steps
          - decaying exploration rate
        """

    @abstractmethod
    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.

        Returns
        -------
        loss : float if an update was performed, None if buffer not ready.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — override in subclass if needed
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist agent state to a .pt checkpoint."""
        raise NotImplementedError(f"{self.__class__.__name__}.save() not implemented")

    def load(self, path: Union[str, Path]) -> None:
        """Restore agent state from a .pt checkpoint."""
        raise NotImplementedError(f"{self.__class__.__name__}.load() not implemented")

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a flat dict of agent diagnostics for the logger.
        Subclasses should call super().get_stats() and update the dict.
        """
        return {
            "agent": self._name,
            "total_steps": self._total_steps,
            "epsilon": self.epsilon,
        }

    def eval_mode(self) -> None:
        """Switch to deterministic (greedy) evaluation mode."""

    def train_mode(self) -> None:
        """Switch back to epsilon-greedy training mode."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, steps={self._total_steps})"
