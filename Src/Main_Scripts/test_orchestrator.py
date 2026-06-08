"""
Orchestrator Simulation Test
=============================
Simulates a complete end-to-end training loop through the AdaptiveTrainingOrchestrator
using the REAL model config and a REAL wrapped model (same as Main.py does),
but feeds FAKE metrics directly into the orchestrator's monitoring pipeline.

This lets you verify:
  1. The metrics are received by the background monitoring thread
  2. The hyperparameter optimizer fires its decision logic correctly
  3. The trainer's adjust_learning_rate() is called with the right value
  4. [DECISION] lines appear in the logs

Run from the Src/Main_Scripts directory:
  python test_orchestrator.py
"""

import sys
import os
import time
import logging
import math
from datetime import datetime

# --- Path Setup (same as Main.py) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'core'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'training'))

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)-7s]  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

print("=" * 70)
print("  ORCHESTRATOR SIMULATION TEST")
print("=" * 70)
print()

# --- Imports (identical to Main.py boot sequence) ---
try:
    from Main import ConfigPresets, config_to_deepseek_config, Config
    log.info("Main.py imports OK")
except Exception as e:
    log.error(f"Cannot import from Main.py: {e}")
    raise

try:
    from training.orchestrator import (
        AdaptiveTrainingOrchestrator,
        TrainingMetrics,
        wrap_orchestrator_with_oom_protection,
    )
    log.info("Orchestrator imports OK")
except Exception as e:
    log.error(f"Cannot import orchestrator: {e}")
    raise

# ============================================================
# PHASE 0 - Build the SAME config as Main.py
# ============================================================
print("\n[PHASE 0] Building real config (debug preset)...")
config = ConfigPresets.debug()

# Mirror the exact same overrides that Main.py applies
config.use_moe         = True
config.use_mod         = True
config.learning_rate   = 3e-4
config.min_lr          = 1e-6
config.warmup_ratio    = 0.1
config.lr_scheduler    = "cosine"
config.num_epochs      = 20
config.batch_size      = 30
config.gradient_accumulation_steps = 8
config.precision       = "mixed_fp16"

# Force verbose decision logging so we see every event
config.log_lr_decisions      = True
config.enable_adaptive_lr    = True
config.min_override_threshold = 0.05   # Lower to 5% so small adjustments show up in test
config.verbosity             = 'normal'

log.info(f"Config: lr={config.learning_rate:.2e}, min_lr={config.min_lr:.2e}, "
         f"warmup={config.warmup_ratio}, scheduler={config.lr_scheduler}")

# ============================================================
# PHASE 1 - Build the orchestrator (same as Main.py Step 10)
# ============================================================
print("\n[PHASE 1] Initializing orchestrator...")
try:
    orchestrator = AdaptiveTrainingOrchestrator(config)
    orchestrator.initialize_training()
    log.info("Orchestrator initialized successfully")
except Exception as e:
    log.error(f"Orchestrator init failed: {e}")
    raise

# ============================================================
# PHASE 2 - Setup scheduler (same as Main.py Step 10.8)
# ============================================================
FAKE_DATASET_SIZE = 115_380   # Same as your real dataset
steps_per_epoch   = FAKE_DATASET_SIZE // (config.batch_size * config.gradient_accumulation_steps)
total_steps       = steps_per_epoch * config.num_epochs
warmup_steps      = int(total_steps * config.warmup_ratio)

print(f"\n[PHASE 2] Setting up scheduler...")
print(f"  Dataset size  : {FAKE_DATASET_SIZE:,} samples")
print(f"  Steps/epoch   : {steps_per_epoch}")
print(f"  Total steps   : {total_steps}")
print(f"  Warmup steps  : {warmup_steps}")

orchestrator.trainer._setup_scheduler(total_steps)
if orchestrator.trainer.scheduler:
    log.info(f"Scheduler created: {type(orchestrator.trainer.scheduler).__name__}")
else:
    log.warning("Scheduler is None - check config.use_lr_scheduler")

# ============================================================
# Utility: build a TrainingMetrics (mirrors the training loop)
# ============================================================
def make_metrics(step: int, loss: float, grad_norm: float, lr: float) -> TrainingMetrics:
    return TrainingMetrics(
        epoch=max(1, step // steps_per_epoch),
        step=step,
        loss=loss,
        grad_norm=grad_norm,
        learning_rate=lr,
        expert_utilization={'expert_0': 0.4, 'expert_1': 0.35,
                            'expert_2': 0.1, 'expert_3': 0.15},
        memory_usage={'gpu_memory_percent': 65.0, 'cpu_percent': 40.0},
        throughput=52000.0,
        semantic_coherence=0.5,
        factual_accuracy=0.4,
        reasoning_score=0.3,
        timestamp=datetime.now(),
    )

# ============================================================
# PHASE 3 - Inject fake metrics directly into _process_real_time_metrics
#           (exactly what the background thread does in production)
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 3: SIMULATED TRAINING SCENARIOS")
print("=" * 70)

lr_log = []  # Track lr changes: (step, old_lr, new_lr, reason)

# Patch adjust_learning_rate to capture calls
original_adjust = orchestrator.trainer.adjust_learning_rate
def patched_adjust(new_lr, grace_period=10):
    old_lr = orchestrator.trainer.optimizer.param_groups[0]['lr']
    original_adjust(new_lr, grace_period=grace_period)
    lr_log.append((orchestrator.global_step, old_lr, new_lr))
    print(f"\n  *** LR CHANGE DETECTED ***  {old_lr:.3e} -> {new_lr:.3e} "
          f"at step {orchestrator.global_step}")
orchestrator.trainer.adjust_learning_rate = patched_adjust


def inject_steps(label, loss_fn, n_steps=120, start_step=0):
    """Feed n_steps of fake metrics through the orchestrator."""
    print(f"\n--- {label} ---")
    for i in range(n_steps):
        step = start_step + i
        lr   = orchestrator.trainer.optimizer.param_groups[0]['lr']
        loss = loss_fn(i)
        gn   = 2.5 + 0.5 * math.sin(i / 10)
        m    = make_metrics(step, loss, gn, lr)

        # This is the EXACT call the background thread makes in production
        orchestrator._process_real_time_metrics(m)

        # Advance the scheduler one step (mirrors trainer.py)
        if orchestrator.trainer.scheduler and not getattr(orchestrator.trainer, '_adaptive_lr_override', False):
            orchestrator.trainer.scheduler.step()

        if (i + 1) % 30 == 0:
            current_lr = orchestrator.trainer.optimizer.param_groups[0]['lr']
            print(f"  Step {step:>4} | Loss={loss:.4f} | LR={current_lr:.3e}")

    return start_step + n_steps


# ---- SCENARIO A: Normal warmup decline (steps 0-119) ----
#  Expected: no interventions (loss dropping naturally)
step = inject_steps(
    label="SCENARIO A: Normal warmup + strong decline (expect NO interventions)",
    loss_fn=lambda i: 10.8 * math.exp(-i / 30),
    n_steps=120,
    start_step=0
)

# ---- SCENARIO B: Plateau (steps 120-239) ----
#  Expected: Orchestrator detects rel_std < 0.001 -> INCREASE LR by 1.5x
step = inject_steps(
    label="SCENARIO B: Hard loss plateau at 5.0 (expect INCREASE intervention)",
    loss_fn=lambda i: 5.0 + 0.0001 * math.sin(i),   # essentially flat
    n_steps=120,
    start_step=step
)

# ---- SCENARIO C: Divergence (steps 240-359) ----
#  Expected: recent_mean > older_mean * 1.1 -> DECREASE LR by 0.5x
step = inject_steps(
    label="SCENARIO C: Loss diverging (expect DECREASE intervention)",
    loss_fn=lambda i: 5.0 + i * 0.15,   # climbing ~10% every ~7 steps
    n_steps=120,
    start_step=step
)

# ---- SCENARIO D: Steady progress (steps 360-479) ----
#  Expected: recent_mean < older_mean * 0.95 -> INCREASE LR by 1.1x
step = inject_steps(
    label="SCENARIO D: Steady strong progress (expect ACCELERATION)",
    loss_fn=lambda i: 4.8 * (0.97 ** i),   # drops ~3% per step
    n_steps=120,
    start_step=step
)

# ---- SCENARIO E: High gradient norms (steps 480-599) ----
#  Expected: mean grad_norm > 10 -> DECREASE LR by 0.7x
print("\n--- SCENARIO E: High gradient norms (expect STABILITY intervention) ---")
for i in range(120):
    step_i = step + i
    lr   = orchestrator.trainer.optimizer.param_groups[0]['lr']
    m    = make_metrics(step_i, 4.5, 12.0 + i * 0.1, lr)
    orchestrator._process_real_time_metrics(m)
    if orchestrator.trainer.scheduler and not getattr(orchestrator.trainer, '_adaptive_lr_override', False):
        orchestrator.trainer.scheduler.step()
    if (i + 1) % 30 == 0:
        current_lr = orchestrator.trainer.optimizer.param_groups[0]['lr']
        print(f"  Step {step_i:>4} | GradNorm=12+ | LR={current_lr:.3e}")
step += 120

# ============================================================
# PHASE 4 - Report
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 4: RESULTS SUMMARY")
print("=" * 70)
print(f"\n  Total steps simulated : {step}")
print(f"  LR changes detected   : {len(lr_log)}")

if lr_log:
    print("\n  LR Change Log:")
    for s, old, new in lr_log:
        direction = "INCREASE" if new > old else "DECREASE"
        print(f"    Step {s:>4}: {old:.3e} -> {new:.3e}  ({direction})")
else:
    print("\n  WARNING: No LR changes were detected!")
    print("  Check the following:")
    print("    1. Is config.enable_adaptive_lr = True?")
    print("    2. Are the min_override_threshold conditions met?")
    print("    3. Is _process_real_time_metrics being called?")

status = orchestrator.get_adaptive_status()
print(f"\n  Orchestrator status:")
print(f"    Adaptive decisions made : {status.get('adaptive_decisions_made', 0)}")
print(f"    Metrics collected       : {status.get('metrics_collected', 0)}")
print(f"    LR adjustments          : {status.get('lr_adjustments', 0)}")

print()
if len(lr_log) >= 3:
    print("  RESULT: PASS - Orchestrator is triggering interventions correctly!")
elif len(lr_log) > 0:
    print("  RESULT: PARTIAL - Some interventions fired, but not all scenarios triggered.")
    print("  Double check the threshold settings and step counts.")
else:
    print("  RESULT: FAIL - No interventions were detected. The orchestrator is broken.")

print("=" * 70)
orchestrator.cleanup()
