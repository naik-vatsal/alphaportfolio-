"""
bandit.py
---------
Contextual UCB1 bandit for selecting among AlphaPortfolio specialists.

Context
-------
Market volatility regime: "low" | "med" | "high"
Computed from recent price returns (see Portfolio._volatility_context).

The bandit maintains separate exploration/exploitation statistics per
context so that, for example, MomentumAgent may be consistently
preferred in low-volatility trending markets while MeanReversionAgent
earns higher selection probability in high-volatility regimes.

UCB1 formula
------------
    score_i = mu_i + c * sqrt(ln(t_ctx) / n_i_ctx)

    mu_i      : mean reward of arm i in current context
    c         : exploration constant (default 2.0)
    t_ctx     : total pulls in current context
    n_i_ctx   : pulls of arm i in current context

Each arm is guaranteed at least one pull per context before UCB scoring
begins (initialisation phase).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


CONTEXTS = ("low", "med", "high")


class ContextualUCBBandit:
    """
    UCB1 bandit with per-context statistics.

    Parameters
    ----------
    n_agents    : number of arms (one per specialist agent)
    agent_names : list of agent name strings for logging
    c           : UCB exploration constant (higher → more exploration)
    contexts    : tuple of context labels (default: low/med/high volatility)
    """

    def __init__(
        self,
        n_agents: int,
        agent_names: List[str],
        c: float = 2.0,
        contexts: tuple = CONTEXTS,
    ) -> None:
        if n_agents != len(agent_names):
            raise ValueError("n_agents must match len(agent_names)")

        self.n_agents = n_agents
        self.agent_names = list(agent_names)
        self.c = c
        self.contexts = contexts

        # Per-context arm statistics
        # _counts[ctx][i]  : number of times arm i was pulled in context ctx
        # _rewards[ctx][i] : cumulative reward of arm i in context ctx
        self._counts: Dict[str, List[int]] = {
            ctx: [0] * n_agents for ctx in contexts
        }
        self._rewards: Dict[str, List[float]] = {
            ctx: [0.0] * n_agents for ctx in contexts
        }
        # Total pulls per context
        self._t: Dict[str, int] = {ctx: 0 for ctx in contexts}

        # Full pull history for analysis
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select(self, context: str) -> int:
        """
        Select an agent index for the given context.

        During initialisation (some arm has n=0), cycles through unpulled
        arms in order.  After all arms have been pulled at least once,
        selects the arm with the highest UCB score.

        Returns
        -------
        arm index in [0, n_agents)
        """
        self._validate_context(context)
        counts = self._counts[context]

        # Initialisation: ensure every arm is tried at least once
        for i, n in enumerate(counts):
            if n == 0:
                return i

        return self._ucb_argmax(context)

    def update(self, agent_idx: int, reward: float, context: str) -> None:
        """
        Record the outcome of a pull.

        Parameters
        ----------
        agent_idx : arm that was pulled
        reward    : observed reward (portfolio return after the step)
        context   : volatility context string
        """
        self._validate_context(context)
        self._counts[context][agent_idx] += 1
        self._rewards[context][agent_idx] += reward
        self._t[context] += 1

        self._history.append(
            {
                "agent": self.agent_names[agent_idx],
                "context": context,
                "reward": reward,
                "t_ctx": self._t[context],
            }
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def ucb_scores(self, context: str) -> Dict[str, float]:
        """Return current UCB scores keyed by agent name."""
        self._validate_context(context)
        counts = self._counts[context]
        t = self._t[context]
        scores: Dict[str, float] = {}

        for i, name in enumerate(self.agent_names):
            n = counts[i]
            if n == 0:
                scores[name] = float("inf")
            else:
                mu = self._rewards[context][i] / n
                conf = self.c * math.sqrt(math.log(max(t, 1)) / n)
                scores[name] = mu + conf

        return scores

    def mean_rewards(self, context: Optional[str] = None) -> Dict:
        """
        Return mean reward per agent, optionally filtered by context.
        If context is None, returns a nested dict {ctx: {agent: mean}}.
        """
        if context is not None:
            self._validate_context(context)
            return {
                self.agent_names[i]: (
                    self._rewards[context][i] / self._counts[context][i]
                    if self._counts[context][i] > 0
                    else 0.0
                )
                for i in range(self.n_agents)
            }
        return {ctx: self.mean_rewards(ctx) for ctx in self.contexts}

    def selection_counts(self) -> Dict[str, Dict[str, int]]:
        """Return {context: {agent_name: count}} pull counts."""
        return {
            ctx: {self.agent_names[i]: self._counts[ctx][i]
                  for i in range(self.n_agents)}
            for ctx in self.contexts
        }

    def get_stats(self) -> Dict:
        return {
            "t": dict(self._t),
            "mean_rewards": self.mean_rewards(),
            "selection_counts": self.selection_counts(),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ucb_argmax(self, context: str) -> int:
        counts = self._counts[context]
        t = self._t[context]
        best_i, best_score = 0, -float("inf")

        for i in range(self.n_agents):
            n = counts[i]
            mu = self._rewards[context][i] / n
            conf = self.c * math.sqrt(math.log(t) / n)
            score = mu + conf
            if score > best_score:
                best_score = score
                best_i = i

        return best_i

    def _validate_context(self, context: str) -> None:
        if context not in self.contexts:
            raise ValueError(
                f"Unknown context '{context}'. Valid: {self.contexts}"
            )
