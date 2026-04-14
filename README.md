# AlphaPortfolio

Multi-agent reinforcement learning system for portfolio management.

## Architecture (planned)

```
alphaportfolio/
├── config.py                  # All hyperparameters
├── environment/
│   ├── market_env.py          # Custom Gymnasium env
│   └── data_loader.py         # yfinance data pipeline
├── agents/                    # (Session 2+)
├── training/                  # (Session 2+)
├── utils/
│   └── logger.py              # Experiment logging
└── runs/                      # Experiment outputs (gitignored)
```

## Quickstart

```bash
pip install -r requirements.txt
python -c "from environment.data_loader import DataLoader; dl = DataLoader(); dl.load(); print(dl.get_split('train')[0].shape)"
```

## Environment

- **State**: `[return, volume, RSI, MACD, MACD_signal, BB_position, BB_width]` × N stocks + portfolio weights + cash ratio
- **Action**: `MultiDiscrete([3] * N)` — 0=sell, 1=hold, 2=buy per stock
- **Reward**: daily portfolio return − transaction cost penalty
- **Episode**: 252 trading days (1 year), random start within the split

## Default Tickers

`AAPL, MSFT, GOOGL, JPM, GLD`

## Sessions

| Session | Scope |
|---------|-------|
| 1 | Env, data pipeline, config, logger |
| 2 | Agent architectures (DQN / PPO) |
| 3 | Training loop, evaluation |
| 4 | LLM regime detection (`USE_LLM_REGIME=True`) |
| 5 | Backtesting & benchmarks |
