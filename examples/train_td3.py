"""
examples/train_td3.py — the SAME environment, a DIFFERENT agent (TD3).

Compared to train_sac.py the only substantive change is SAC -> TD3.
That is the point of a Gymnasium environment: the problem stays fixed, and
any algorithm can be dropped in to learn a policy for it.

HEADLESS, and produces the SAME artifacts as the SAC script:
    harness_td3.zip
    training_harness_td3.log
    learning_curve_harness_td3.png

Run:  python examples/train_td3.py
"""

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from train_common import run_training

if __name__ == "__main__":
    # TD3 is deterministic, so it needs explicit exploration noise
    action_noise = NormalActionNoise(mean=np.zeros(3), sigma=0.1 * np.ones(3))

    run_training(
        TD3, tag="harness_td3",
        total_episodes=500,
        max_ep_steps=300,
        detail_every=25,
        checkpoint_every=50,
        learning_rate=1e-3, buffer_size=200_000, batch_size=256, gamma=0.99,
        action_noise=action_noise,
    )
