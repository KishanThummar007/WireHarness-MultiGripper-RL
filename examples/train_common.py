"""
=============================================================================
 examples/train_common.py  —  shared training machinery
=============================================================================
Both train_sac.py and train_td3.py import this, so BOTH get identical
episode logging, a training.log file, and a learning-curve graph.

Training is HEADLESS (no MuJoCo window) — that is deliberate:
  * rendering would slow training by orders of magnitude
  * it lets you train on a remote server / cluster over SSH with no display
The trained policy is saved as a .zip, which you then load in watch.py to
SEE it in the MuJoCo viewer. Train headless -> watch locally.
=============================================================================
"""

import logging
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback

from harness_rl import make_env


def build_logger(log_file):
    logger = logging.getLogger("harness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w"); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


class EpisodeLogger(BaseCallback):
    """Logs each episode (PASS/FAIL, reward, distance), optionally each step,
    checkpoints, stops after N episodes, and records data for the graph."""

    def __init__(self, logger, total_episodes, detail_every, checkpoint_every, tag):
        super().__init__()
        self.log = logger
        self.total_episodes = total_episodes
        self.detail_every = detail_every
        self.checkpoint_every = checkpoint_every
        self.tag = tag
        self.ep = 0
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.rewards_hist, self.success_hist, self.dist_hist = [], [], []

    def _on_step(self) -> bool:
        action = self.locals["actions"][0]
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        self.ep_reward += reward
        self.ep_steps += 1

        if self.detail_every and (self.ep % self.detail_every == 0):
            g, t = info["grip"], info["target"]
            self.log.info(
                f"  ep{self.ep:04d} s{self.ep_steps:4d} "
                f"act[{action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f}] "
                f"grip[{g[0]:.2f} {g[1]:.2f} {g[2]:.2f}] "
                f"tgt[{t[0]:.2f} {t[1]:.2f} {t[2]:.2f}] "
                f"dist {info['distance']:.3f} rew {reward:+.3f}")

        if done:
            success = bool(info.get("is_success", False))
            g = info["grip"]
            self.log.info(
                f"EPISODE {self.ep:04d} | steps {self.ep_steps:4d} "
                f"| reward {self.ep_reward:+8.2f} | {'PASS' if success else 'FAIL'} "
                f"| final dist {info['distance']:.3f} "
                f"| grip[{g[0]:.2f} {g[1]:.2f} {g[2]:.2f}]")

            self.rewards_hist.append(self.ep_reward)
            self.success_hist.append(1 if success else 0)
            self.dist_hist.append(float(info["distance"]))

            self.ep += 1
            self.ep_reward = 0.0
            self.ep_steps = 0

            if self.checkpoint_every and self.ep % self.checkpoint_every == 0:
                self.model.save(f"{self.tag}_ep{self.ep}")
                self.log.info(f"  [checkpoint] saved {self.tag}_ep{self.ep}.zip")

            if self.ep >= self.total_episodes:
                self.log.info(f"reached {self.total_episodes} episodes -> stopping")
                return False
        return True


def plot_curve(logger, rewards, success, dists, path, title):
    import matplotlib
    matplotlib.use("Agg")          # headless backend (no display needed)
    import matplotlib.pyplot as plt

    def rolling(x, k=10):
        x = np.asarray(x, float)
        return x if len(x) < k else np.convolve(x, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    axes[0].plot(rewards, alpha=0.35, label="per episode")
    axes[0].plot(rolling(rewards), lw=2, label="rolling mean (10)")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("total reward")
    axes[0].set_title(f"{title} — learning curve"); axes[0].legend()

    axes[1].plot(rolling(success, 10) * 100, lw=2, color="green")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("success rate (%)")
    axes[1].set_title("Rolling success rate (10 ep)"); axes[1].set_ylim(-5, 105)

    axes[2].plot(dists, alpha=0.35, color="darkred")
    axes[2].plot(rolling(dists), lw=2, color="red")
    axes[2].set_xlabel("episode"); axes[2].set_ylabel("final distance (m)")
    axes[2].set_title("Final distance to connector (should trend DOWN)")

    fig.tight_layout(); fig.savefig(path, dpi=120)
    logger.info(f"saved learning curve -> {path}")


def run_training(agent_cls, tag, total_episodes=500, max_ep_steps=300,
                 detail_every=25, checkpoint_every=50, **agent_kwargs):
    """Register the env, build the agent, train with logging + graph."""
    logger = build_logger(f"training_{tag}.log")

    register(id="HarnessPick-v0", entry_point=lambda: make_env(),
             max_episode_steps=max_ep_steps)
    env = gym.make("HarnessPick-v0")

    agent = agent_cls("MlpPolicy", env, verbose=0, **agent_kwargs)

    cb = EpisodeLogger(logger, total_episodes, detail_every, checkpoint_every, tag)
    logger.info(f"=== training {tag.upper()} for {total_episodes} episodes "
                f"(max {max_ep_steps} steps/ep, headless) ===")
    agent.learn(total_timesteps=total_episodes * max_ep_steps, callback=cb)

    agent.save(tag)
    logger.info(f"saved final policy -> {tag}.zip  (load it in watch.py to SEE it)")
    plot_curve(logger, cb.rewards_hist, cb.success_hist, cb.dist_hist,
               f"learning_curve_{tag}.png", tag.upper())
