"""
test_session4.py
----------------
End-to-end smoke test for Session 4: training loop + evaluation pipeline.

Runs Experiment 1 (baseline DQN) for 5 specialist episodes and 5
orchestrator episodes only — enough to exercise every code path without
waiting for real convergence.

Checks:
  1. run_experiment_1 completes without error
  2. Metrics dict contains all expected keys for every strategy
  3. Learning curve PNG was written to disk
  4. JSON results file was written to disk
"""

from pathlib import Path
import sys

EXPECTED_METRIC_KEYS = {
    "cumulative_return", "sharpe_ratio", "max_drawdown",
    "win_rate", "final_value", "n_steps",
}

# ── Run Experiment 1 ─────────────────────────────────────────────────
print("=" * 60)
print("SESSION 4 SMOKE TEST — Experiment 1 (5 episodes)")
print("=" * 60)

from experiments.run_all import run_experiment_1

result = run_experiment_1(
    n_specialist_episodes=5,
    n_orchestrator_episodes=5,
)

# ── Metrics table ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("METRICS TABLE")
print("=" * 60)

metrics = result["metrics"]
assert isinstance(metrics, dict) and len(metrics) > 0, "metrics dict is empty"

header = f"{'Strategy':<30}  {'CumRet':>8}  {'Sharpe':>7}  {'MaxDD':>8}  {'WinRate':>8}"
print(header)
print("-" * len(header))

for strategy, m in metrics.items():
    missing = EXPECTED_METRIC_KEYS - set(m.keys())
    assert not missing, f"{strategy} missing keys: {missing}"
    print(
        f"{strategy:<30}  "
        f"{m['cumulative_return']:>+8.4f}  "
        f"{m['sharpe_ratio']:>7.3f}  "
        f"{m['max_drawdown']:>8.4f}  "
        f"{m['win_rate']:>8.3f}"
    )

# ── File checks ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FILE CHECKS")
print("=" * 60)

results_dir = Path("experiments") / "results" / "exp1_baseline_dqn"

plot_path = results_dir / "learning_curve.png"
assert plot_path.exists(), f"Learning curve plot not found: {plot_path}"
print(f"Learning curve plot : {plot_path}  ({plot_path.stat().st_size:,} bytes)  OK")

json_path = results_dir / "metrics.json"
assert json_path.exists(), f"Metrics JSON not found: {json_path}"
print(f"Metrics JSON        : {json_path}  ({json_path.stat().st_size:,} bytes)  OK")

checkpoint_files = list(results_dir.parent.parent.glob("**/checkpoints/*.pt"))
print(f"Checkpoint files    : {len(checkpoint_files)} .pt file(s) saved")

# ── Specialist stats sanity checks ───────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING STATS SANITY")
print("=" * 60)

specialist_stats = result["result"]["specialist_stats"]
for agent_name, stats_list in specialist_stats.items():
    assert len(stats_list) == 5, f"{agent_name}: expected 5 episode records, got {len(stats_list)}"
    final = stats_list[-1]
    print(
        f"{agent_name:<28}  "
        f"ep={final['episode']}  "
        f"ret={final['episode_return']:+.5f}  "
        f"eps={final.get('epsilon', 0):.3f}  "
        f"loss={final.get('mean_loss', 0):.6f}"
    )

orch_stats = result["result"]["orchestrator_stats"]
assert len(orch_stats) == 5, f"orchestrator: expected 5 records, got {len(orch_stats)}"
final_orch = orch_stats[-1]
print(
    f"{'orchestrator':<28}  "
    f"ep={final_orch['episode']}  "
    f"ret={final_orch['episode_return']:+.5f}"
)

# ── Done ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL SESSION 4 CHECKS PASSED — end-to-end pipeline functional.")
print("=" * 60)
