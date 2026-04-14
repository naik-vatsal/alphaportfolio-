from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    TICKERS: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "JPM", "GLD"])
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2024-12-31"
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    # TEST_RATIO is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.15

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    N_STOCKS: int = 5
    INITIAL_CASH: float = 1_000_000.0
    TRANSACTION_COST: float = 0.001   # 10 bps per trade (applied to notional)
    EPISODE_LENGTH: int = 252         # 1 trading year

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    # Features per stock: return, volume, RSI, MACD, MACD_signal, BB_position, BB_width
    FEATURES_PER_STOCK: int = 7

    # ------------------------------------------------------------------
    # RL hyperparameters
    # ------------------------------------------------------------------
    HIDDEN_DIM: int = 256               # Q-network hidden layer width
    USE_DOUBLE_DQN: bool = True         # Double DQN target computation
    COORD_ALPHA: float = 0.7            # local-vs-global reward mixing (coordinator)
    LEARNING_RATE: float = 3e-4
    GAMMA: float = 0.99               # discount factor
    EPSILON_START: float = 1.0        # exploration ε-greedy start
    EPSILON_END: float = 0.05         # exploration ε-greedy floor
    EPSILON_DECAY: int = 100_000      # steps over which ε is annealed
    BATCH_SIZE: int = 128
    REPLAY_BUFFER_SIZE: int = 100_000
    TARGET_UPDATE_FREQ: int = 1_000   # steps between hard target-net updates
    TAU: float = 0.005                # soft update coefficient (Polyak)
    N_STEP_RETURN: int = 3            # n-step TD return

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    MAX_EPISODES: int = 2_000
    EVAL_FREQ: int = 50               # episodes between validation runs
    CHECKPOINT_FREQ: int = 100        # episodes between checkpoints
    SEED: int = 42

    # ------------------------------------------------------------------
    # LLM regime detection  (off by default — Session 4)
    # ------------------------------------------------------------------
    USE_LLM_REGIME: bool = False
    LLM_MODEL: str = "claude-opus-4-6"
    LLM_REGIME_INTERVAL: int = 5      # trading days between LLM calls

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_DIR: str = "runs"
    EXPERIMENT_NAME: str = "alphaportfolio_v1"


# Module-level singleton — import this everywhere
CFG = Config()
