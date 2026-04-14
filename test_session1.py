"""
test_session1.py
----------------
Smoke test for Session 1 components: DataLoader → MarketEnv pipeline.
No training — just verifies data flows through correctly.
"""

import sys
import numpy as np

# ── 1. DataLoader ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: DataLoader")
print("=" * 60)

from environment.data_loader import DataLoader
from config import CFG

loader = DataLoader()
loader.load()

train_features, train_prices = loader.get_split("train")
val_features,   val_prices   = loader.get_split("val")
test_features,  test_prices  = loader.get_split("test")

print(f"\nTickers      : {CFG.TICKERS}")
print(f"Train split  : features={train_features.shape}  prices={train_prices.shape}")
print(f"Val split    : features={val_features.shape}  prices={val_prices.shape}")
print(f"Test split   : features={test_features.shape}  prices={test_prices.shape}")

print("\nFirst 3 rows of training features:")
header = [f"{t}_{f}" for t in CFG.TICKERS
          for f in ["ret", "vol", "rsi", "macd", "macd_sig", "bb_pos", "bb_width"]]
col_w = 11
print("  " + "".join(f"{h:>{col_w}}" for h in header))
for i, row in enumerate(train_features[:3]):
    print(f"t={i} " + "".join(f"{v:>{col_w}.4f}" for v in row))

# ── 2. MarketEnv ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: MarketEnv — instantiation")
print("=" * 60)

from environment.market_env import MarketEnv

env = MarketEnv(
    features=train_features,
    close_prices=train_prices,
    seed=CFG.SEED,
)

print(f"\nObservation space : {env.observation_space}")
print(f"Action space      : {env.action_space}")
print(f"  obs_dim breakdown: {train_features.shape[1]} tech features "
      f"+ {CFG.N_STOCKS} stock weights + 1 cash weight "
      f"= {env.observation_space.shape[0]}")

# ── 3. reset() ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: env.reset()")
print("=" * 60)

obs, info = env.reset(seed=CFG.SEED)

print(f"\nobs shape         : {obs.shape}  dtype={obs.dtype}")
print(f"obs[:5] (tech)    : {obs[:5]}")
print(f"obs[-6:] (weights): {obs[-6:]}  (5 stocks + cash, sum~={obs[-6:].sum():.4f})")
print(f"portfolio_value   : ${info['portfolio_value']:,.2f}")
print(f"cash              : ${info['cash']:,.2f}")
print(f"episode start t   : {info['timestep']}")

# ── 4. step() with random action ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: env.step() — 5 random actions")
print("=" * 60)

rng = np.random.default_rng(CFG.SEED)
total_reward = 0.0

print(f"\n{'Step':>5}  {'Action':<20}  {'Reward':>10}  {'Port. Value':>14}  {'TC':>10}")
print("-" * 65)

for step in range(5):
    action = env.action_space.sample()
    action_str = str(["SELL","HOLD","BUY"][a] for a in action).replace("<generator object <genexpr>", "")
    action_labels = [["SELL","HOLD","BUY"][a] for a in action]
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"{step+1:>5}  {str(action_labels):<20}  {reward:>+10.6f}  "
          f"${info['portfolio_value']:>13,.2f}  ${info['transaction_cost']:>9.2f}")

print(f"\nTotal reward over 5 steps : {total_reward:+.6f}")
print(f"Terminated                : {terminated}  Truncated: {truncated}")
print(f"obs shape after step      : {obs.shape}")

# ── 5. render() ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: env.render()")
print("=" * 60)
print()
env.render()

# ── 6. Full episode length check ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Full episode termination check")
print("=" * 60)

obs, info = env.reset()
steps = 0
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    done = terminated or truncated

print(f"\nEpisode ran for {steps} steps (expected {CFG.EPISODE_LENGTH})")
assert steps == CFG.EPISODE_LENGTH, f"Expected {CFG.EPISODE_LENGTH} steps, got {steps}"
print("Termination check PASSED.")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL CHECKS PASSED — Session 1 pipeline is functional.")
print("=" * 60)
