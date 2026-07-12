"""
examples/train_sac.py — train with SAC (Soft Actor-Critic).

HEADLESS: no MuJoCo window is opened (fast; works over SSH on a server).
Produces:
    harness_sac.zip                 the trained policy -> load in watch.py to SEE it
    training_harness_sac.log        per-episode PASS/FAIL + per-step detail
    learning_curve_harness_sac.png  reward / success-rate / distance-vs-radius graphs

Run:  python examples/train_sac.py
"""

from stable_baselines3 import SAC
from train_common import run_training

if __name__ == "__main__":
    run_training(
        SAC, tag="harness_sac",
        total_episodes=500,      # raise once the success rate is climbing
        max_ep_steps=500,        # 300 was too short: episodes reached 0.2-0.3 m and RAN OUT
                                 #   of time before the scripted handoff could finish the grasp
        detail_every=25,
        checkpoint_every=50,
        # curriculum: start with a generous scripted-handoff radius (config.py = 0.80)
        # and tighten it whenever the agent succeeds in >=70% of the last 10 episodes
        curriculum=True, curric_window=10, curric_thresh=0.70,
        curric_step=0.10, curric_floor=0.15,
        learning_rate=3e-4, buffer_size=200_000, batch_size=256, gamma=0.99,
    )
