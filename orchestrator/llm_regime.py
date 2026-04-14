"""
llm_regime.py
-------------
LLM-powered market regime detector using the Claude API.

Active only when config.USE_LLM_REGIME = True.  When disabled (default),
the detector silently falls back to a pure volatility heuristic so the
rest of the pipeline requires no conditional logic.

Regime labels (canonical)
--------------------------
  "trending"        — consistent directional price movement
  "mean_reverting"  — oscillation around a mean, no persistent trend
  "volatile"        — large erratic moves, no clear pattern
  "uncertain"       — insufficient signal; model is not confident

Claude API
----------
  Model  : claude-haiku-4-5-20251001  (fast, cheap, appropriate for structured classification)
  Input  : last 5 trading days of OHLCV-derived statistics
  Output : exactly one of the four regime labels above

Caching
-------
  API responses are cached in a fixed-size LRU dict keyed on an MD5
  hash of the last-5-day closing prices (rounded to 2 dp).
  Cache is in-process only (not persisted to disk).

Fallback
--------
  Any API failure (network error, auth error, unexpected response) is
  caught and the heuristic detector is used instead.  A warning is
  printed so the user knows the fallback was triggered.

Installation note
-----------------
  The `anthropic` package must be installed:
      pip install anthropic>=0.25.0
  Add it to requirements.txt before enabling USE_LLM_REGIME.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional

import numpy as np

from config import CFG

# Optional import — only required when USE_LLM_REGIME = True
try:
    import anthropic as _anthropic_module
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


VALID_REGIMES = frozenset({"trending", "mean_reverting", "volatile", "uncertain"})
_CACHE_MAX_SIZE = 512


class LLMRegimeDetector:
    """
    Detects the current market regime from recent price data.

    Parameters
    ----------
    model       : Claude model ID (default: claude-haiku-4-5-20251001)
    cache_size  : maximum number of cached LLM responses
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        cache_size: int = _CACHE_MAX_SIZE,
    ) -> None:
        self.model = model
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size
        self._client = None        # lazy-initialised on first LLM call
        self._llm_calls: int = 0
        self._fallback_calls: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        features: np.ndarray,
        prices: np.ndarray,
    ) -> str:
        """
        Classify the current market regime.

        Parameters
        ----------
        features : float32 array (T, n_stocks * FEATURES_PER_STOCK)
                   Normalised technical features from DataLoader.
        prices   : float32 array (T, n_stocks) — raw closing prices

        Returns
        -------
        One of: "trending", "mean_reverting", "volatile", "uncertain"
        """
        if not CFG.USE_LLM_REGIME:
            return self._heuristic(prices)

        if not _ANTHROPIC_AVAILABLE:
            print(
                "[LLMRegime] WARNING: `anthropic` package not installed. "
                "Install it with: pip install anthropic>=0.25.0\n"
                "           Falling back to heuristic regime detection."
            )
            return self._heuristic(prices)

        cache_key = self._cache_key(prices)
        if cache_key in self._cache:
            # Move to end (most-recently used)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        try:
            regime = self._call_llm(prices)
        except Exception as exc:
            print(f"[LLMRegime] API call failed ({type(exc).__name__}: {exc}). "
                  "Falling back to heuristic.")
            self._fallback_calls += 1
            regime = self._heuristic(prices)

        self._put_cache(cache_key, regime)
        return regime

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, prices: np.ndarray) -> str:
        if self._client is None:
            self._client = _anthropic_module.Anthropic()  # reads ANTHROPIC_API_KEY

        prompt = self._build_prompt(prices)

        message = self._client.messages.create(
            model=self.model,
            max_tokens=16,          # regime label is at most ~16 chars
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip().lower()

        # Extract the first valid regime word found in the response
        for regime in VALID_REGIMES:
            if regime in raw:
                self._llm_calls += 1
                return regime

        # Model returned something unexpected — treat as uncertain
        print(f"[LLMRegime] Unexpected response: '{raw}'. Defaulting to 'uncertain'.")
        self._llm_calls += 1
        return "uncertain"

    def _build_prompt(self, prices: np.ndarray) -> str:
        """
        Build a concise structured prompt from the last 5 days of prices.
        Keeps token usage minimal for haiku.
        """
        window = prices[-min(5, len(prices)):]          # (≤5, n_stocks)
        n_days, n_stocks = window.shape

        # Day-over-day returns as percentage
        if n_days >= 2:
            rets = np.diff(window, axis=0) / (window[:-1] + 1e-8) * 100
            rets_rounded = np.round(rets, 2).tolist()
        else:
            rets_rounded = []

        # Summary stats
        mean_ret = float(np.mean(rets)) if rets_rounded else 0.0
        vol = float(np.std(rets)) if rets_rounded else 0.0
        trend_score = float(np.mean(np.sign(rets[-1]))) if rets_rounded else 0.0

        prompt = (
            f"You are a quantitative market analyst. Classify the market regime "
            f"for a portfolio of {n_stocks} stocks using the last {n_days} trading days.\n\n"
            f"Day-over-day returns (%, columns = stocks):\n{rets_rounded}\n\n"
            f"Summary: mean_return={mean_ret:.3f}%  volatility={vol:.3f}%  "
            f"trend_score={trend_score:.2f}\n\n"
            f"Respond with EXACTLY ONE word from this list:\n"
            f"trending | mean_reverting | volatile | uncertain\n\n"
            f"Rules:\n"
            f"- trending: consistent directional movement across most stocks\n"
            f"- mean_reverting: oscillation, alternating signs, low net movement\n"
            f"- volatile: large erratic moves (|return| > 2%) without clear direction\n"
            f"- uncertain: mixed signals or insufficient data\n\n"
            f"Your answer (one word only):"
        )
        return prompt

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic(self, prices: np.ndarray) -> str:
        """
        Volatility- and trend-based regime classification.
        Used when USE_LLM_REGIME=False or as API fallback.
        """
        if len(prices) < 5:
            return "uncertain"

        window = prices[-20:] if len(prices) >= 20 else prices
        rets = np.diff(window, axis=0) / (window[:-1] + 1e-8)  # (T-1, n_stocks)

        vol = float(np.std(rets))
        mean_ret = float(np.mean(rets))
        # Trend: fraction of days where sign is consistent with overall direction
        signs = np.sign(rets)
        trend_consistency = float(np.abs(np.mean(signs)))

        if vol > 0.025:
            return "volatile"
        if trend_consistency > 0.65:
            return "trending"
        if vol < 0.010 and trend_consistency < 0.40:
            return "mean_reverting"
        return "uncertain"

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, prices: np.ndarray) -> str:
        window = prices[-5:] if len(prices) >= 5 else prices
        rounded = np.round(window, 2)
        return hashlib.md5(rounded.tobytes()).hexdigest()

    def _put_cache(self, key: str, value: str) -> None:
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)  # evict oldest
        self._cache[key] = value

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "llm_calls": self._llm_calls,
            "fallback_calls": self._fallback_calls,
            "cache_size": len(self._cache),
            "use_llm": CFG.USE_LLM_REGIME,
        }

    def __repr__(self) -> str:
        return (
            f"LLMRegimeDetector(model={self.model!r}, "
            f"use_llm={CFG.USE_LLM_REGIME}, "
            f"llm_calls={self._llm_calls}, "
            f"cached={len(self._cache)})"
        )
