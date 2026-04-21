# Reinforcement Learning Approach

AlphaPortfolio layers three distinct RL mechanisms that operate at different
timescales and levels of abstraction. This document covers their mathematical
formulations, design rationale, hyperparameter choices, and the empirical
finding that motivated the warmup stabilization phase.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Approach 1 — Deep Q-Network (DQN)](#2-approach-1--deep-q-network-dqn)
3. [Approach 2 — Contextual UCB Bandit](#3-approach-2--contextual-ucb-bandit)
4. [Approach 3 — Cooperative Reward Sharing (MARL)](#4-approach-3--cooperative-reward-sharing-marl)
5. [Three-Phase Training Protocol](#5-three-phase-training-protocol)
6. [Warmup Stabilization Finding](#6-warmup-stabilization-finding)
7. [Hyperparameter Reference](#7-hyperparameter-reference)

---

## 1. Problem Formulation

Portfolio management is modelled as a discrete-time Markov Decision Process
(MDP) with the following structure.

**State** `s_t ∈ ℝ^41`

```
s_t = [ tech_features (35) | stock_weights (5) | cash_weight (1) ]
```

The 35 technical features are 7 z-scored indicators per stock:
daily return, volume, RSI-14, MACD, MACD signal, Bollinger Band position,
and Bollinger Band width. Portfolio weights and the cash fraction sum to 1
and are appended so the agent observes its own allocation.

**Action** `a_t ∈ {0,1,2}^5`

Each element is a per-stock decision: 0 = sell, 1 = hold, 2 = buy.
The environment translates these signals into proportional capital reallocation
with equal buy/sell pressure per stock.

**Reward** `r_t`

```
r_t = (V_{t+1} - V_t) / V_t  −  TC_t
```

where `V_t` is portfolio value at step `t` and `TC_t = 0.001 × notional_traded`
is a 10 bps transaction cost applied to the total dollar value of position
changes. The transaction cost term penalizes excessive trading and encourages
the agent to form convictions before acting.

**Episode** — 252 trading days (one calendar year) with a random start within
the training split, resampled at the beginning of every episode.

---

## 2. Approach 1 — Deep Q-Network (DQN)

### 2.1 Architecture

Each specialist agent maintains two copies of a 3-layer MLP Q-network:

```
Q(s, a ; θ) :   obs_dim → 256 → 256 → (n_stocks × n_actions)
```

The output of shape `(n_stocks × n_actions)` is reshaped to `(n_stocks, n_actions)`
so that per-stock Q-values can be computed and actions selected independently
for each asset. Weights are initialized with orthogonal initialization
(gain = √2 for hidden layers, gain = 0.01 for the output layer) to produce
stable Q-values at the start of training.

### 2.2 Double DQN Loss

Standard DQN suffers from overestimation bias because the same network
selects and evaluates the greedy next action. Double DQN decouples these
two operations:

```
a*_{t+1} = argmax_a  Q_online(s_{t+1}, a ; θ)        [online net selects]
y_t      = r_t + γ^n · Q_target(s_{t+1}, a*_{t+1} ; θ⁻)  [target net evaluates]
```

The loss minimized at each gradient step is:

```
L(θ) = E_{(s,a,r,s') ~ D} [ ( Q_online(s, a ; θ) − y )² ]
```

where `D` is the replay buffer and the expectation is approximated over a
minibatch of size 128. Gradients are clipped to a maximum L2 norm of 10.0
before the Adam update.

**Why Double DQN?** In portfolio management, overestimated Q-values translate
directly to overconfident position-taking. Double DQN reduces this bias without
additional compute, which is important when the reward signal is inherently
noisy (daily returns have high variance relative to any single agent's edge).

### 2.3 N-Step Returns

Rather than storing single-step `(s_t, a_t, r_t, s_{t+1})` tuples, the
`NStepBuffer` wrapper accumulates n transitions and computes the discounted
sum before pushing to the replay buffer:

```
G_{t:t+n} = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{n-1}·r_{t+n-1}
```

The stored tuple becomes `(s_t, a_t, G_{t:t+n}, s_{t+n}, done_{t+n})` and
the target uses `γ^n · Q_target(s_{t+n}, ...)` to account for the longer
bootstrap horizon. At episode boundaries, the pending window is flushed with
whatever partial returns are available so no experience is discarded.

**Why n=3?** Single-step TD has high variance in financial environments because
individual daily returns are dominated by noise. n=3 spreads credit assignment
over three trading days — a window that captures short-term price reactions
to position changes — while avoiding the high bias that comes with very long
Monte Carlo returns.

### 2.4 Target Network Updates

The target network `θ⁻` is updated every `TARGET_UPDATE_FREQ = 1000` gradient
steps via a hard copy:

```
θ⁻  ←  θ   (every 1000 steps)
```

Soft Polyak averaging is also implemented but not used in the default training
loop:

```
θ⁻  ←  τ · θ + (1 − τ) · θ⁻    (τ = 0.005)
```

### 2.5 Epsilon-Greedy Exploration

Exploration follows a linear annealing schedule:

```
ε(t) = max( ε_end,  ε_start − t · (ε_start − ε_end) / ε_decay )
```

with `ε_start = 1.0`, `ε_end = 0.05`, and `ε_decay = 100 000` steps.
During evaluation, `agent.eval_mode()` forces `ε = 0` (fully greedy).

### 2.6 Specialist Hyperparameter Overrides

Each specialist agent inherits `DQNAgent` and overrides a subset of
hyperparameters to match its intended market regime. The global `CFG`
singleton is never mutated; overrides are applied through a lightweight
proxy (`_CfgOverride`) that intercepts attribute lookups.

| Agent | γ (gamma) | ε_end | Hidden dim | Rationale |
|---|---|---|---|---|
| MomentumAgent | 0.995 | 0.02 | 256 | Trends unfold over weeks; high γ weights distant future rewards heavily. Decisive policy (low ε_end). |
| MeanReversionAgent | 0.97 | 0.08 | 512 | Reversion plays out in days; lower γ focuses on near-term rewards. Higher ε_end maintains exploratory flexibility. Wider net to capture non-linear oscillation patterns. |
| MacroAgent | 0.99 (default) | 0.05 (default) | 256 (default) | Trained on all conditions; serves as the ensemble anchor. No regime-specific tuning. |

**Segment filters** — each specialist exposes a `segment_filter(features, prices)`
predicate that the orchestrator can query to determine whether a price window
is suitable for that specialist:

- `MomentumAgent`: triggers when mean absolute 21-day cross-sectional return
  exceeds 0.6 (strong directional movement present)
- `MeanReversionAgent`: triggers when recent 5-day volatility exceeds 1.5×
  the full-window baseline std (elevated oscillatory activity)
- `MacroAgent`: always returns `True` (unconditional training anchor)

---

## 3. Approach 2 — Contextual UCB Bandit

### 3.1 Purpose

The bandit is the orchestrator's routing layer. At each environment step it
selects which of the three specialist agents proposes the action. The key
insight is that agent suitability is not stationary — a momentum strategy that
works well in a trending market may underperform in a volatile one. The bandit
learns per-context selection policies without requiring explicit regime labels.

### 3.2 Volatility Context

Recent price volatility maps to one of three bandit contexts based on the
trailing 20-day standard deviation of portfolio daily returns:

```
context =  "low"   if σ_20 < 0.010  (< 1.0% annualised daily std)
           "med"   if 0.010 ≤ σ_20 ≤ 0.020
           "high"  if σ_20 > 0.020  (> 2.0%)
```

Separate arm statistics are maintained per context so that, for example,
`MomentumAgent` can accumulate high selection counts in `"low"` volatility
without distorting the `"high"` volatility statistics.

### 3.3 UCB1 Formula

For each (context, arm) pair, the UCB score is:

```
score_i(ctx) = μ_i(ctx) + c · √( ln(t_ctx) / n_i(ctx) )
```

where:
- `μ_i(ctx) = Σ rewards_i(ctx) / n_i(ctx)` — mean reward of arm i in context ctx
- `t_ctx` — total pulls in context ctx across all arms
- `n_i(ctx)` — pulls of arm i in context ctx
- `c = 2.0` — exploration constant

The arm with the highest score is selected. During initialization, each arm
is guaranteed at least one pull per context (in index order) before UCB scoring
begins, preventing the logarithm from diverging and ensuring all agents receive
early exposure.

**Why UCB over Thompson Sampling or ε-greedy bandit?**
UCB provides deterministic, interpretable selection and has rigorous O(√T log T)
regret bounds. In a portfolio context this is preferable to stochastic selection
(Thompson Sampling) because the training loop benefits from reproducible
agent routing during debugging and ablation studies. The exploration bonus
also decays naturally as data accumulates, providing automatic annealing
without a separate schedule.

**Why c=2.0?** The standard UCB1 theory uses c=√2 ≈ 1.41. We use c=2.0 to
encourage slightly more exploration during the relatively short orchestrator
training budget (500–2000 episodes). With ~1500 steps per episode and three
agents, each agent receives on the order of 500 000 pulls over a full run,
at which point the confidence bonus is small regardless of c.

### 3.4 Bandit Update

After each environment step the orchestrator calls:

```python
bandit.update(agent_idx, blended_reward, context)
```

The reward passed to the bandit is the coordinator-blended reward (see
Section 4), not the raw environment reward. This ensures the bandit's arm
statistics reflect cooperative performance rather than individual agent returns,
aligning bandit optimization with the team objective.

---

## 4. Approach 3 — Cooperative Reward Sharing (MARL)

### 4.1 Motivation

Naively applying independent DQN to a multi-agent portfolio system creates
two problems:

1. **Non-stationarity** — as all agents update simultaneously, the effective
   environment appears non-stationary from each agent's perspective, since the
   routing policy (bandit) changes which agent's actions take effect.

2. **Credit assignment** — the agent that was *not* selected at step t still
   needs a learning signal. Without reward sharing, non-selected agents receive
   no gradient information for most of Phase 2.

Cooperative reward sharing addresses both by giving every agent a signal every
step, weighted by peer confidence.

### 4.2 Reward Blending Formula

At each step, every agent receives a blended reward:

```
r_blended_i = α · r_local_i + (1 − α) · r_global
```

where the global signal is a confidence-weighted average across all agents:

```
r_global = Σ_j  [ conf_j / Σ_k conf_k ] · r_local_j
```

and confidence is the maximum Q-value of agent j's online network at the
current observation:

```
conf_j = max_a  Q_online_j(s_t, a)
```

All agents in this system receive the same local reward (`r_local_i = r_env`
for all i) because only one agent's action is executed, and the environment
reward is attributed to the team rather than to the selected individual.

**Full expansion for α = 0.7:**

```
r_blended_i = 0.7 · r_env  +  0.3 · Σ_j (conf_j / Σ_k conf_k) · r_env
            = r_env · [ 0.7  +  0.3 · Σ_j (conf_j / Σ_k conf_k) ]
            = r_env        (when all confidences are equal)
```

When confidences differ, the global term upweights agents that are more
certain about the current state, allowing high-confidence agents to
disproportionately shape the shared learning signal.

### 4.3 Alpha (α) Design Decision

α controls the local-vs-global trade-off:

| α | Behavior | Risk |
|---|---|---|
| 1.0 | Pure local reward — no sharing | Non-selected agents receive no gradient; collapses to independent DQN performance |
| 0.7 | Default mix | Agents retain individual incentive while benefiting from peer signal |
| 0.0 | Pure global (full cooperation) | All agents receive identical signal; individual specialization is diluted |

The Experiment 5 ablation confirms this ordering empirically:

```
α = 1.0  →  Sharpe 0.80   (≈ random baseline)
α = 0.0  →  Sharpe 1.74
α = 0.7  →  Sharpe 2.23   ← optimal
```

**Why not tune α continuously?** A fixed α keeps the training dynamics
stationary and interpretable. Dynamic α schedules (e.g., annealing from 0.5
to 0.7 as agents mature) are a natural extension but were not explored within
the scope of this project.

### 4.4 Q-Value Confidence as a Communication Channel

Using max-Q as a confidence proxy is a lightweight alternative to explicit
agent communication protocols. It requires no additional network parameters
and is differentiable in principle, though it is used here only for weighting
the reward signal. The intuition is that an agent with high max-Q has a
strong opinion about the current state and should contribute more to the
team signal than an agent whose Q-values are flat (uncertain).

A practical edge case: early in training, Q-values are small and approximately
equal across agents. In this regime `conf_j / Σ_k conf_k ≈ 1/N` for all j, so
the global reward reduces to a simple average — a reasonable uninformed prior.

---

## 5. Three-Phase Training Protocol

```
Phase 1  ──────────────────────────────────────
  Each specialist trains independently on the training split.
  No bandit, no coordinator.
  Duration: n_specialist_episodes per agent (default 200).
  Purpose: give each agent a competent base policy before
           the orchestrator begins routing.

Phase 1.5  (Warmup)  ──────────────────────────
  Each specialist runs independently *in the orchestrated
  environment* (same obs/action space as Phase 2).
  No bandit routing; each agent drives all steps solo.
  Duration: n_warmup_episodes per agent (default 0; recommended 200).
  Purpose: populate the replay buffer with on-distribution
           transitions before multi-agent training begins.

Phase 2  ──────────────────────────────────────
  Full system: bandit routes steps, coordinator blends rewards,
  all agents update every step.
  Duration: n_orchestrator_episodes (default 500).
  Purpose: joint optimization of the team objective.
```

The warmup phase is optional (`n_warmup_episodes=0` disables it) and adds
negligible wall-clock time relative to Phase 2 at recommended episode budgets.

---

## 6. Warmup Stabilization Finding

### 6.1 The Cold-Start Problem

When Phase 2 begins immediately after Phase 1, the UCB bandit has no
accumulated statistics and routes steps uniformly at random (initialization
phase). During these early episodes:

- All three agents receive highly variable routing (each selected ~33% of steps)
- The blended reward is noisy because no agent has yet built reliable Q-value
  estimates in the new multi-agent observation distribution
- Replay buffers — which were populated during solo Phase 1 — contain
  transitions from a different distribution than Phase 2 produces (solo env
  dynamics differ slightly from orchestrated env dynamics when reward blending
  is active)

The result is a period of instability in early Phase 2 gradients that can
prevent the bandit from learning a useful routing policy.

### 6.2 Evidence from Experiment Results

The effect is clearly visible across experiments:

| System | Sharpe Ratio | vs Buy-and-Hold |
|---|---|---|
| DQN + UCB, no coord, no warmup (Exp 2) | 0.80 | −65% |
| DQN + UCB + coord, no warmup (Exp 3) | 2.23 | −2% |
| Full system + warmup (Exp 4) | 2.31 | +2% |

Experiment 2 is the clearest demonstration: despite having the same specialist
agents as Experiment 3, the absence of cooperative reward sharing leaves the
bandit with no stabilizing signal during its initialization phase. The resulting
Sharpe of 0.80 is statistically indistinguishable from the random agent (0.78).

Experiments 3 and 4 both use reward coordination, which acts as a partial
substitute for warmup: the global reward component provides a consistent signal
even when the bandit is routing randomly, because all agents update every step
and their joint Q-value confidence converges faster than solo agents would.

### 6.3 Mechanism of Warmup Stabilization

The warmup phase (`train_warmup` in `training/train.py`) addresses the
cold-start problem by running each agent through `n_warmup_episodes` episodes
in the orchestrated environment before the bandit begins routing:

```python
for agent in portfolio.agents:
    for ep in range(n_warmup_episodes):
        obs, _ = env.reset()
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
```

This achieves three things:

1. **Buffer pre-population** — each agent's replay buffer contains on-distribution
   transitions from the Phase 2 environment (same observation space, same
   reward structure) before any bandit routing occurs.

2. **Q-value calibration** — agents adapt their Q-value estimates to the
   new reward distribution (which may differ from Phase 1 if reward blending
   was active during Phase 1; it is not by default, but the transition cost
   structure and episode dynamics are identical).

3. **Reduced bandit initialization variance** — when Phase 2 begins, agents
   have calibrated Q-values so the confidence-weighted global reward is
   immediately more informative, leading to faster bandit convergence.

### 6.4 Recommended Usage

```bash
# Minimum recommended warmup for 500-episode orchestrator runs
python -m experiments.run_all --exp 3 --spec-ep 500 --orch-ep 500 --warmup-ep 100

# Standard configuration used in Experiments 3 and 4
python -m experiments.run_all --exp 4 --spec-ep 500 --orch-ep 2000 --warmup-ep 200
```

At 200 warmup episodes × 252 steps × 3 agents = ~151 200 additional transitions,
each agent's replay buffer (capacity 100 000) is effectively pre-filled before
Phase 2 begins. Warmup episodes beyond ~400 show diminishing returns because
the buffer cycles completely and early transitions are overwritten.

---

## 7. Hyperparameter Reference

### DQN Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| `HIDDEN_DIM` | 256 | Sufficient capacity for a 41-dimensional state with 5-stock portfolio; larger than necessary for the tabular features, but allows future extension to raw price sequences |
| `LEARNING_RATE` | 3e-4 | Standard Adam learning rate for DQN; lower rates slow convergence, higher rates cause Q-value instability |
| `GAMMA` | 0.99 | Balances immediate vs. future portfolio returns; 0.99 gives ~100-step effective horizon, appropriate for 252-step episodes |
| `BATCH_SIZE` | 128 | Standard DQN batch size; larger batches reduce gradient variance but slow per-step wall time |
| `REPLAY_BUFFER_SIZE` | 100 000 | Stores ~400 episodes of experience; large enough to decorrelate consecutive transitions, small enough to remain in memory |
| `EPSILON_START` | 1.0 | Full random exploration at the start ensures diverse buffer population |
| `EPSILON_END` | 0.05 | 5% exploration floor prevents policy collapse in non-stationary market conditions |
| `EPSILON_DECAY` | 100 000 | Steps over which ε anneals; spans ~400 training episodes at 252 steps each |
| `TARGET_UPDATE_FREQ` | 1 000 | Hard update every 1000 gradient steps; frequent enough to track the online net, infrequent enough to provide stable targets |
| `TAU` | 0.005 | Polyak soft-update coefficient (implemented but not used in default training) |
| `N_STEP_RETURN` | 3 | 3-step returns propagate credit over 3 trading days; balances bias-variance trade-off |
| `USE_DOUBLE_DQN` | True | Reduces overestimation bias at no cost; always beneficial in noisy reward environments |
| `GRAD_CLIP` | 10.0 | L2 gradient clipping prevents exploding gradients during early training |

### Bandit Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| `UCB_C` | 2.0 | Exploration constant; slightly above the theoretical √2 to encourage exploration at short episode budgets |
| Contexts | low / med / high | Three volatility contexts sufficient to capture the main market regimes; adding more contexts reduces data per context and slows bandit convergence |
| Vol thresholds | 1% / 2% daily std | Approximately the 33rd and 67th percentile of realized daily volatility in the 2015–2024 dataset; confirmed by inspecting the data split distribution |

### Coordinator Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| `COORD_ALPHA` | 0.7 | Ablation (Exp 5) confirms this is optimal across the three tested values; gives agents enough local signal to maintain specialization while sharing enough global signal to stabilize learning |

### Training Schedule

| Parameter | Default | Notes |
|---|---|---|
| `n_specialist_episodes` | 200 | Minimum for agents to leave the random-walk regime; 500 recommended for production runs |
| `n_warmup_episodes` | 0 | 200 recommended whenever Phase 2 episodes > 500 |
| `n_orchestrator_episodes` | 500 | 2000 recommended for convergence; Sharpe improvements plateau around 1500 episodes in practice |
| `checkpoint_freq` | 50 | Save every 50 episodes; frequent enough to recover from crashes without excessive I/O |
