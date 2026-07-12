"""
harness_rl — a Gymnasium environment for multi-gripper wire-harness picking (MuJoCo).

Importing this package registers the environment ID, so users can do:

    import gymnasium as gym
    import harness_rl                  # registers "HarnessPick-v0"
    env = gym.make("HarnessPick-v0")

Reward terms live in env.py -> HarnessPickEnv._compute_reward().
The RL agent (SAC / TD3 / PPO) is separate — see examples/.
"""

from gymnasium.envs.registration import register

from .env import HarnessPickEnv, save_snapshot, S0_SPEC
from .config import make_env

register(
    id="HarnessPick-v0",
    entry_point="harness_rl.config:make_env",
    max_episode_steps=600,
)

__all__ = ["HarnessPickEnv", "save_snapshot", "S0_SPEC", "make_env"]
