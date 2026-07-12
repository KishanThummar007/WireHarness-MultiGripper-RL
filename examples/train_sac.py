"""
examples/train_sac.py — train with SAC (Soft Actor-Critic).

HEADLESS: no MuJoCo window is opened (fast; works over SSH on a server).
Produces:
    harness_sac.zip            the trained policy  -> load in watch.py to SEE it
    training_harness_sac.log   per-episode + per-step log
    learning_curve_harness_sac.png  reward / success-rate / distance graphs

Run:  python examples/train_sac.py
"""

from stable_baselines3 import SAC
from train_common import run_training

if __name__ == "__main__":
    run_training(
        SAC, tag="harness_sac",
        total_episodes=500,      # raise once you see the curves trending correctly
        max_ep_steps=300,        # the gripper needs far fewer than 600 steps now
        detail_every=25,
        checkpoint_every=50,
        learning_rate=3e-4, buffer_size=200_000, batch_size=256, gamma=0.99,
    )
