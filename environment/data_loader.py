"""
data_loader.py
--------------
Downloads OHLCV data via yfinance, computes technical features, and
splits the resulting feature matrix into train / val / test without
any lookahead bias.

Normalisation uses only training-set statistics (μ, σ) so the same
scaler is applied consistently to val and test.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import CFG


# ──────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ──────────────────────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _macd(
    prices: pd.Series, fast: int, slow: int, signal: int
) -> Tuple[pd.Series, pd.Series]:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bollinger(
    prices: pd.Series, period: int, n_std: float
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
        bb_position : (price − MA) / (n_std × σ)  — roughly in [-1, 1]
        bb_width    : (upper − lower) / MA          — normalised band width
    """
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    bb_position = ((prices - ma) / (n_std * std + 1e-8)).fillna(0.0)
    bb_width = ((2 * n_std * std) / (ma + 1e-8)).fillna(0.0)
    return bb_position, bb_width


def _build_ticker_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 technical features for a single ticker."""
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    ret = close.pct_change().fillna(0.0)
    # Volume relative to its 20-day average (> 1 means above-average activity)
    vol_norm = (volume / volume.rolling(20).mean().replace(0, np.nan)).fillna(1.0)

    rsi = _rsi(close, CFG.RSI_PERIOD) / 100.0  # scale to [0, 1]

    macd_line, macd_sig = _macd(close, CFG.MACD_FAST, CFG.MACD_SLOW, CFG.MACD_SIGNAL)
    # Normalise MACD by price so it is scale-invariant across tickers
    macd_norm = (macd_line / (close + 1e-8)).fillna(0.0)
    macd_sig_norm = (macd_sig / (close + 1e-8)).fillna(0.0)

    bb_pos, bb_width = _bollinger(close, CFG.BB_PERIOD, CFG.BB_STD)

    return pd.DataFrame(
        {
            f"{ticker}_ret": ret,
            f"{ticker}_vol": vol_norm,
            f"{ticker}_rsi": rsi,
            f"{ticker}_macd": macd_norm,
            f"{ticker}_macd_sig": macd_sig_norm,
            f"{ticker}_bb_pos": bb_pos,
            f"{ticker}_bb_width": bb_width,
        },
        index=df.index,
    )


# ──────────────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Fetches, processes, and time-splits market data for AlphaPortfolio.

    Usage
    -----
    >>> loader = DataLoader()
    >>> loader.load()
    >>> train_features, train_prices = loader.get_split("train")
    >>> val_features,   val_prices   = loader.get_split("val")
    >>> test_features,  test_prices  = loader.get_split("test")
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        self.tickers: List[str] = tickers or CFG.TICKERS
        self.start: str = start or CFG.START_DATE
        self.end: str = end or CFG.END_DATE

        self._raw: Dict[str, pd.DataFrame] = {}
        self._features: Optional[pd.DataFrame] = None
        self._close_prices: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download raw data and build the normalised feature matrix."""
        self._download()
        self._build_feature_matrix()

    def get_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (features, close_prices) for the requested split.

        Parameters
        ----------
        split : "train" | "val" | "test"

        Returns
        -------
        features     : float32 ndarray of shape (T, n_stocks × FEATURES_PER_STOCK)
        close_prices : float32 ndarray of shape (T, n_stocks)
        """
        self._require_loaded()
        sl = self._split_slice(split)
        return (
            self._features.values[sl].astype(np.float32),
            self._close_prices.values[sl].astype(np.float32),
        )

    def get_dates(self, split: str) -> pd.DatetimeIndex:
        """Return the DatetimeIndex corresponding to the requested split."""
        self._require_loaded()
        return self._features.index[self._split_slice(split)]

    @property
    def n_stocks(self) -> int:
        return len(self.tickers)

    @property
    def feature_dim(self) -> int:
        return self.n_stocks * CFG.FEATURES_PER_STOCK

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self) -> None:
        print(f"[DataLoader] Downloading {len(self.tickers)} tickers "
              f"({self.start} to {self.end}) ...")
        for ticker in self.tickers:
            df = yf.download(
                ticker,
                start=self.start,
                end=self.end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                raise ValueError(f"yfinance returned no data for '{ticker}'")
            self._raw[ticker] = df
        print(f"[DataLoader] Download complete.")

    def _build_feature_matrix(self) -> None:
        """Align tickers on common trading days, compute features, z-score normalise."""
        # Intersection of all trading calendars to avoid NaN rows
        common_index: Optional[pd.DatetimeIndex] = None
        for df in self._raw.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)

        all_features: List[pd.DataFrame] = []
        close_dict: Dict[str, pd.Series] = {}

        for ticker in self.tickers:
            df = self._raw[ticker].loc[common_index]
            all_features.append(_build_ticker_features(ticker, df))
            close_dict[ticker] = df["Close"].squeeze()

        feature_df = pd.concat(all_features, axis=1).dropna()
        close_df = pd.DataFrame(close_dict).loc[feature_df.index]

        # Z-score using ONLY training-set statistics (no lookahead)
        n = len(feature_df)
        train_end = int(n * CFG.TRAIN_RATIO)
        mu = feature_df.iloc[:train_end].mean()
        sigma = feature_df.iloc[:train_end].std().replace(0.0, 1.0)
        feature_df = (feature_df - mu) / sigma

        self._features = feature_df
        self._close_prices = close_df

        val_len = int(n * CFG.VAL_RATIO)
        test_len = n - train_end - val_len
        print(
            f"[DataLoader] Feature matrix: {feature_df.shape}  |  "
            f"train={train_end}  val={val_len}  test={test_len}"
        )

    def _split_slice(self, split: str) -> slice:
        n = len(self._features)
        train_end = int(n * CFG.TRAIN_RATIO)
        val_end = train_end + int(n * CFG.VAL_RATIO)
        slices = {
            "train": slice(0, train_end),
            "val":   slice(train_end, val_end),
            "test":  slice(val_end, n),
        }
        if split not in slices:
            raise ValueError(f"split must be one of {list(slices)}, got '{split}'")
        return slices[split]

    def _require_loaded(self) -> None:
        if self._features is None:
            raise RuntimeError("Call load() before accessing split data.")
