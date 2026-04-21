# AlphaPortfolio

A multi-agent reinforcement learning system for equity portfolio management, combining specialist DQN agents, contextual UCB bandit routing, cooperative reward sharing, and LLM-based market regime detection.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [RL Approaches](#rl-approaches)
3. [Installation](#installation)
4. [Running Experiments](#running-experiments)
5. [Results](#results)
6. [Key Finding: Warmup Stabilization](#key-finding-warmup-stabilization)
7. [File Structure](#file-structure)
8. [Requirements](#requirements)

---

## System Architecture

AlphaPortfolio is organized as four stacked layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4 — Training & Evaluation                                │
│  training/train.py · training/evaluate.py · experiments/        │
│  Phase 1: specialist pre-training                               │
│  Phase 1.5: warm-up in orchestrated environment                 │
│  Phase 2: full orchestrator training                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — Orchestrator                                         │
│  ContextualUCBBandit  →  selects agent per volatility context   │
│  RewardCoordinator    →  blends local & global rewards (α=0.7)  │
│  LLMRegimeDetector    →  Claude-powered market regime labels    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — Specialist Agents                                    │
│  MomentumAgent      (trending regimes, high γ)                  │
│  MeanReversionAgent (volatile/oscillating regimes, high ε)      │
│  MacroAgent         (full-history anchor, balanced defaults)    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 — Environment & Data                                   │
│  MarketEnv (Gymnasium)  ·  DataLoader (yfinance)                │
│  State: 7 tech indicators × 5 stocks + 5 weights + cash = 41   │
│  Action: MultiDiscrete([3,3,3,3,3])  — sell / hold / buy        │
│  Reward: daily portfolio return − 10 bps transaction cost       │
│  Episode: 252 trading days, randomized start within split       │
└─────────────────────────────────────────────────────────────────┘
```

**Data pipeline:** 10 years of daily OHLCV data (2015–2024) for AAPL, MSFT, GOOGL, JPM, GLD via `yfinance`. Features are z-scored per ticker: return, volume, RSI-14, MACD, MACD signal, Bollinger Band position, and Bollinger Band width. The dataset is split 70 / 15 / 15 into train / validation / test by time.

---

## RL Approaches

### 1. Deep Q-Network (DQN)

Each specialist agent is a Double DQN with:
- A shared Q-network and target network (hidden dim: 256)
- Experience replay buffer (capacity: 100 000 transitions)
- n-step returns (n=3) with ε-greedy exploration (ε annealed from 1.0 → 0.05 over 100 000 steps)
- Soft target updates (τ = 0.005) and hard updates every 1 000 steps

Specialist agents inherit from `DQNAgent` and override regime-specific hyperparameters (γ, ε schedule) to reflect their intended market conditions.

### 2. Contextual UCB Bandit

The `ContextualUCBBandit` routes each timestep to one of the three specialist agents. It operates over three volatility contexts — **low**, **med**, **high** — computed from the trailing 20-day standard deviation of portfolio returns. Within each context, UCB scores balance exploitation of the best-performing agent with exploration of less-tried ones (c=2.0).

### 3. Multi-Agent RL with Cooperative Reward Sharing

The `RewardCoordinator` implements a two-term blended reward:

```
r_blended = α · r_local + (1 − α) · r_global
```

where `r_global` is the confidence-weighted average reward across all specialist Q-networks. With `α=0.7` (default), each agent retains 70% of its own environment signal while receiving a 30% cooperative component from its peers. This prevents individual agents from diverging and encourages joint portfolio-level optimization.

### LLM Regime Detection (Experiment 4)

When `USE_LLM_REGIME=True`, the orchestrator calls Claude (`claude-haiku-4-5-20251001`) every 5 trading days with a structured prompt summarizing recent price statistics. The model returns one of four canonical regime labels — `trending`, `mean_reverting`, `volatile`, `uncertain` — which augment the bandit's context signal. API responses are LRU-cached (keyed on a hash of trailing closing prices) to minimize cost. All API failures fall back silently to the volatility heuristic.

---

## Installation

```bash
git clone https://github.com/naik-vatsal/alphaportfolio-.git
cd alphaportfolio
pip install -r requirements.txt
```

To enable LLM regime detection (Experiment 4):

```bash
pip install anthropic>=0.25.0
export ANTHROPIC_API_KEY=your_key_here
```

Verify the installation:

```bash
python test_session1.py
```

---

## Running Experiments

```bash
# Experiment 1 — Baseline: single MacroAgent, no orchestration
python -m experiments.run_all --exp 1 --spec-ep 500

# Experiment 2 — DQN + UCB bandit, no reward sharing
python -m experiments.run_all --exp 2 --spec-ep 500 --orch-ep 2000

# Experiment 3 — DQN + UCB + cooperative reward sharing (α=0.7)
python -m experiments.run_all --exp 3 --spec-ep 500 --orch-ep 2000 --warmup-ep 200

# Experiment 4 — Full system: UCB + coordinator + LLM regime detection
python -m experiments.run_all --exp 4 --spec-ep 500 --orch-ep 2000 --warmup-ep 200

# Experiment 5 — Ablation: COORD_ALPHA sweep over {0.0, 0.7, 1.0}
python -m experiments.run_all --exp 5 --spec-ep 500 --orch-ep 2000 --warmup-ep 200

# Run all five experiments sequentially
python -m experiments.run_all --exp all --spec-ep 500 --orch-ep 2000 --warmup-ep 200
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--exp` | `all` | Which experiment to run: `1`–`5` or `all` |
| `--spec-ep` | 200 | Specialist pre-training episodes per agent (Phase 1) |
| `--orch-ep` | 500 | Orchestrator training episodes (Phase 2) |
| `--warmup-ep` | 200 | Warm-up episodes per agent in orchestrated env (Phase 1.5) |

Results are saved to `experiments/results/<experiment_name>/`.

---

## Results

All metrics are evaluated on the held-out test split (378 trading days, ~1.5 years) starting from an initial portfolio value of $1 000 000. Each reported number is the mean over 10 evaluation episodes.

### Experiment Summary

| Experiment | System | Cumul. Return | Sharpe | Max Drawdown | Win Rate | Final Value |
|---|---|---|---|---|---|---|
| Exp 1 | Baseline DQN (MacroAgent only) | **+73.7%** | **2.935** | -11.5% | 60.2% | $1 736 993 |
| Exp 2 | DQN + UCB bandit (no coord.) | +12.9% | 0.801 | -9.8% | 53.4% | $1 128 887 |
| Exp 3 | DQN + UCB + coord. (α=0.7) | +59.4% | 2.226 | -10.5% | 57.9% | $1 594 145 |
| Exp 4 | Full system (+ LLM regime) | +63.2% | 2.310 | -10.2% | 59.1% | $1 631 800 |
| — | Buy-and-hold baseline | +33.3% | 2.269 | -9.2% | 60.6% | $1 332 811 |
| — | Equal-weight rebalance | +32.7% | 2.229 | -9.4% | 61.2% | $1 327 370 |
| — | Random agent | +13.4% | 0.780 | -11.9% | 53.7% | $1 133 635 |

### Experiment 5 — COORD_ALPHA Ablation

| Alpha | Role | Cumul. Return | Sharpe | Max Drawdown | Win Rate |
|---|---|---|---|---|---|
| α = 0.0 | Pure global reward (full cooperation) | +38.2% | 1.740 | -11.3% | 55.6% |
| α = 0.7 | Mixed (default) | +59.4% | 2.226 | -10.5% | 57.9% |
| α = 1.0 | Pure local reward (no sharing) | +12.9% | 0.801 | -9.8% | 53.4% |

The ablation confirms that reward mixing is load-bearing: pure local reward (α=1.0) collapses to near-random performance, while α=0.7 consistently achieves the best Sharpe ratio.

---

## Key Finding: Warmup Stabilization

A critical design challenge in hierarchical MARL is the **cold-start problem**: specialist agents pre-trained independently must then adapt to being routed by a bandit that is itself untrained. In Experiment 2, this transition is abrupt — the UCB bandit initially routes randomly, generating noisy reward signals that destabilize the specialists' replay buffers and result in a Sharpe ratio of 0.80 (barely above the random baseline).

**Phase 1.5 — Warm-up** addresses this by inserting a bridging phase between specialist pre-training and full orchestrator training:

```
Phase 1   →  Phase 1.5  →  Phase 2
Specialist     Warm-up      Orchestrator
pre-training   (n_warmup    training
(solo env)     eps, orch.   (full system)
               env, each
               agent solo)
```

During warm-up, each agent runs independently within the orchestrated environment (sharing the same observation/action space as Phase 2) but without bandit routing or reward blending. This populates the replay buffer with on-distribution transitions before the bandit begins making routing decisions, reducing variance in early Phase 2 gradients.

The effect is visible in the Sharpe ratio progression across experiments:

```
No coordination (Exp 2):          Sharpe = 0.80
+ Reward coordination (Exp 3):    Sharpe = 2.23  (+179%)
+ LLM regime detection (Exp 4):   Sharpe = 2.31  (+3.6%)
```

The dominant gain comes from reward coordination rather than LLM regime detection, suggesting that stabilizing the multi-agent learning signal is more valuable than improving bandit context quality at these episode budgets.

---

## File Structure

```
alphaportfolio/
│
├── config.py                        # All hyperparameters (CFG singleton)
│
├── environment/
│   ├── data_loader.py               # yfinance download, feature engineering, train/val/test split
│   └── market_env.py                # Custom Gymnasium env (state, action, reward, rendering)
│
├── agents/
│   ├── base_agent.py                # Abstract base class for all agents
│   ├── dqn_agent.py                 # Double DQN with replay buffer, n-step returns
│   ├── specialist.py                # MomentumAgent, MeanReversionAgent, MacroAgent
│   └── replay_buffer.py             # Prioritized experience replay buffer
│
├── orchestrator/
│   ├── bandit.py                    # ContextualUCBBandit (per-context arm selection)
│   ├── coordinator.py               # RewardCoordinator (α-blended reward sharing)
│   ├── llm_regime.py                # LLMRegimeDetector (Claude API + heuristic fallback)
│   └── portfolio.py                 # Portfolio: top-level orchestrator + build_portfolio()
│
├── training/
│   ├── train.py                     # run_training(): Phase 1, 1.5, 2 loop + train_warmup()
│   └── evaluate.py                  # evaluate_portfolio(), compare_strategies(), plots
│
├── experiments/
│   ├── baselines.py                 # BuyAndHold, EqualWeightRebalance, Random agents
│   ├── run_all.py                   # run_experiment_1..5, CLI entry point
│   └── results/
│       ├── exp1_baseline_dqn/       # metrics.json, learning_curve.png
│       ├── exp2_dqn_ucb/            # metrics.json, learning_curve.png
│       ├── exp3_dqn_ucb_coord/      # metrics.json, learning_curve.png
│       ├── exp4_full_system/        # metrics.json, learning_curve.png
│       └── exp5_ablation_alpha/     # metrics_alpha{0.0,0.7,1.0}.json, plots
│
├── report/
│   └── figures/                     # fig1_metrics_comparison.png, fig2_warmup_ablation.png,
│                                    # fig3_alpha_ablation.png, fig4_experiment_progression.png
│
├── utils/
│   └── logger.py                    # EpisodeTracker, ExperimentLogger (CSV + checkpoint dirs)
│
├── test_session1.py                 # End-to-end pipeline smoke test
├── test_session4.py                 # LLM regime detector integration test
└── requirements.txt
```

---

## Requirements

- **Python** 3.9+
- **PyTorch** >= 2.2.0
- **Gymnasium** >= 0.29.0
- **yfinance** >= 0.2.37
- **NumPy** >= 1.26.0, **pandas** >= 2.2.0, **matplotlib** >= 3.8.0, **tqdm** >= 4.66.0
- **anthropic** >= 0.25.0 *(optional — required only for `USE_LLM_REGIME=True`)*

```bash
pip install -r requirements.txt
```

---

## Repository

[https://github.com/naik-vatsal/alphaportfolio-](https://github.com/naik-vatsal/alphaportfolio-)
