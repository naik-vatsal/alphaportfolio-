"""
market_env.py
-------------
Custom Gymnasium environment for multi-stock portfolio management.

Observation space (flat float32 vector):
    [tech_features (n_stocks × FEATURES_PER_STOCK) | stock_weights (n_stocks) | cash_weight (1)]

Action space:
    MultiDiscrete([3] * n_stocks)
    0 = sell all,  1 = hold,  2 = buy (equal allocation)

Reward:
    portfolio_pct_return  −  transaction_cost_fraction

Episode:
    252 steps (1 trading year); start index is sampled uniformly from the split.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import CFG


class MarketEnv(gym.Env):
    """
    Multi-stock portfolio management environment.

    Parameters
    ----------
    features      : float32 array (T, n_stocks × FEATURES_PER_STOCK)
                    Pre-normalised technical features from DataLoader.
    close_prices  : float32 array (T, n_stocks)
                    Closing prices used for trade execution and P&L.
    n_stocks      : number of tradeable assets (default: CFG.N_STOCKS)
    initial_cash  : starting cash (default: CFG.INITIAL_CASH)
    transaction_cost : fraction of notional charged per trade (default: CFG.TRANSACTION_COST)
    episode_length   : steps per episode (default: CFG.EPISODE_LENGTH)
    seed          : optional RNG seed
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        n_stocks: Optional[int] = None,
        initial_cash: Optional[float] = None,
        transaction_cost: Optional[float] = None,
        episode_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.features = features.astype(np.float32)          # (T, n_stocks × F)
        self.close_prices = close_prices.astype(np.float32)  # (T, n_stocks)

        self.n_stocks = n_stocks or CFG.N_STOCKS
        self.initial_cash = initial_cash if initial_cash is not None else CFG.INITIAL_CASH
        self.transaction_cost = transaction_cost if transaction_cost is not None else CFG.TRANSACTION_COST
        self.episode_length = episode_length or CFG.EPISODE_LENGTH

        T = self.features.shape[0]
        assert self.close_prices.shape == (T, self.n_stocks), (
            f"close_prices must be ({T}, {self.n_stocks}), "
            f"got {self.close_prices.shape}"
        )
        assert T > self.episode_length + 1, (
            f"Dataset length {T} must exceed episode_length {self.episode_length}"
        )

        # obs = tech features + portfolio weights + cash weight
        obs_dim = self.features.shape[1] + self.n_stocks + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # 0 = sell, 1 = hold, 2 = buy
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)

        self._rng = np.random.default_rng(seed)

        # Episode state — set properly in reset()
        self._t: int = 0
        self._start_t: int = 0
        self._cash: float = float(self.initial_cash)
        self._holdings: np.ndarray = np.zeros(self.n_stocks, dtype=np.float32)
        self._portfolio_value: float = float(self.initial_cash)

    # ──────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a random start so episodes cover different market regimes
        max_start = len(self.features) - self.episode_length - 1
        self._start_t = int(self._rng.integers(0, max(max_start, 1)))
        self._t = self._start_t

        self._cash = float(self.initial_cash)
        self._holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self._portfolio_value = self._cash

        return self._obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Parameters
        ----------
        action : int array of shape (n_stocks,), values in {0, 1, 2}

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prices_now = self.close_prices[self._t]
        prev_value = self._portfolio_value

        # Execute orders at today's close
        tc = self._execute_trades(action, prices_now)

        # Advance to next timestep
        self._t += 1
        prices_next = self.close_prices[self._t]

        # Mark portfolio to market at next close
        self._portfolio_value = self._cash + float(np.dot(self._holdings, prices_next))

        # Reward: percentage return minus transaction cost fraction
        pct_return = (self._portfolio_value - prev_value) / (prev_value + 1e-8)
        reward = float(pct_return) - tc / (prev_value + 1e-8)

        terminated = (self._t - self._start_t) >= self.episode_length
        truncated = False

        info = self._info()
        info["transaction_cost"] = tc
        info["pct_return"] = pct_return
        return self._obs(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        prices = self.close_prices[self._t]
        weights = self._portfolio_weights(prices)
        lines = [
            f"[MarketEnv] step={self._t - self._start_t}/{self.episode_length}"
            f"  value=${self._portfolio_value:,.2f}",
            f"  cash=${self._cash:,.2f}  ({weights[-1] * 100:.1f}%)",
        ]
        for i in range(self.n_stocks):
            lines.append(
                f"  stock[{i}]: {self._holdings[i]:.4f} sh "
                f"@ ${prices[i]:.2f}  "
                f"(weight {weights[i] * 100:.1f}%)"
            )
        out = "\n".join(lines)
        if mode == "human":
            print(out)
        return out

    def close(self) -> None:
        pass

    # ──────────────────────────────────────────────────────────────────
    # Trade execution
    # ──────────────────────────────────────────────────────────────────

    def _execute_trades(self, action: np.ndarray, prices: np.ndarray) -> float:
        """
        Simple signal-to-order conversion:
          - SELL (0): liquidate full position in that stock.
          - HOLD (1): no change.
          - BUY  (2): allocate an equal share of available cash to each buy signal.

        Returns total transaction cost in dollars.
        """
        total_cost = 0.0

        # Phase 1 — liquidate sell signals
        for i in np.where(action == 0)[0]:
            if self._holdings[i] > 0.0:
                proceeds = float(self._holdings[i] * prices[i])
                cost = proceeds * self.transaction_cost
                self._cash += proceeds - cost
                total_cost += cost
                self._holdings[i] = 0.0

        # Phase 2 — buy signals with equal share of remaining cash
        buy_idx = np.where(action == 2)[0]
        if len(buy_idx) > 0:
            cash_per_stock = self._cash / len(buy_idx)
            for i in buy_idx:
                # Net spend after transaction cost
                net_spend = cash_per_stock / (1.0 + self.transaction_cost)
                shares = net_spend / (prices[i] + 1e-8)
                cost = net_spend * self.transaction_cost
                self._cash -= cash_per_stock
                self._holdings[i] += shares
                total_cost += cost

        return total_cost

    # ──────────────────────────────────────────────────────────────────
    # Observation helpers
    # ──────────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        tech = self.features[self._t]                         # (n_stocks × F,)
        weights = self._portfolio_weights(self.close_prices[self._t])  # (n_stocks + 1,)
        return np.concatenate([tech, weights], dtype=np.float32)

    def _portfolio_weights(self, prices: np.ndarray) -> np.ndarray:
        """Returns [stock_weight_0, ..., stock_weight_N-1, cash_weight]."""
        stock_values = self._holdings * prices
        total = stock_values.sum() + self._cash + 1e-8
        stock_weights = (stock_values / total).astype(np.float32)
        cash_weight = np.array([self._cash / total], dtype=np.float32)
        return np.concatenate([stock_weights, cash_weight])

    def _info(self) -> Dict[str, Any]:
        return {
            "portfolio_value": self._portfolio_value,
            "cash": self._cash,
            "holdings": self._holdings.copy(),
            "timestep": self._t,
            "episode_step": self._t - self._start_t,
        }
