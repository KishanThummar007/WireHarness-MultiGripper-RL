"""
=============================================================================
 examples/train_common.py  —  shared training machinery
=============================================================================
Both train_sac.py and train_td3.py import this, so BOTH get identical
episode logging, a training.log file, learning-curve graphs, and the
CURRICULUM.

Training is HEADLESS (no MuJoCo window) — that is deliberate:
  * rendering would slow training by orders of magnitude
  * it lets you train on a remote server / cluster over SSH with no display
The trained policy is saved as a .zip, which you then load in watch.py to
SEE it in the MuJoCo viewer. Train headless -> watch locally.

CURRICULUM (this is what makes the task learnable):
  The env hands the final approach to a scripted controller once the gripper
  is within `approach_radius` (config.py starts it GENEROUS at 0.80). If that
  radius is too tight, the agent NEVER reaches it, never earns the success
  bonus, and has NO learning signal -> flat 0% success forever.
  Once the agent succeeds reliably, we SHRINK the radius, so the policy is
  pushed from "get roughly close" -> "reach the connector precisely".
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
    """Logs each episode (PASS/FAIL, reward, distance, radius), optionally each
    step, runs the CURRICULUM, checkpoints, stops after N episodes, and records
    data for the graphs."""

    def __init__(self, logger, total_episodes, detail_every, checkpoint_every, tag,
                 curriculum=True, curric_window=10, curric_thresh=0.70,
                 curric_step=0.10, curric_floor=0.15):
        super().__init__()
        self.log = logger
        self.total_episodes = total_episodes
        self.detail_every = detail_every
        self.checkpoint_every = checkpoint_every
        self.tag = tag
        # curriculum settings
        self.curriculum = curriculum
        self.curric_window = curric_window
        self.curric_thresh = curric_thresh
        self.curric_step = curric_step
        self.curric_floor = curric_floor

        self.ep = 0
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.rewards_hist, self.success_hist, self.dist_hist = [], [], []
        self.radius_hist = []

    def _env(self):
        """The environment instance the agent is training on."""
        return self.training_env.envs[0].unwrapped

    def _maybe_tighten(self):
        """If the agent succeeds reliably, make the task harder by shrinking the
        radius at which the scripted controller takes over."""
        if not self.curriculum or len(self.success_hist) < self.curric_window:
            return
        recent = float(np.mean(self.success_hist[-self.curric_window:]))
        if recent < self.curric_thresh:
            return
        env = self._env()
        cur = float(getattr(env, "approach_radius", 0.0))
        new = max(self.curric_floor, cur - self.curric_step)
        if new < cur - 1e-9:
            env.approach_radius = new
            self.log.info(f"  [curriculum] success {recent*100:.0f}% "
                          f">= {self.curric_thresh*100:.0f}%  ->  approach_radius "
                          f"{cur:.2f} -> {new:.2f}   (task just got harder)")
            # reset the window so the agent must re-earn success at the new radius
            self.success_hist[-self.curric_window:] = [0] * self.curric_window

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
            radius = float(getattr(self._env(), "approach_radius", 0.0))
            self.log.info(
                f"EPISODE {self.ep:04d} | steps {self.ep_steps:4d} "
                f"| reward {self.ep_reward:+8.2f} | {'PASS' if success else 'FAIL'} "
                f"| final dist {info['distance']:.3f} | radius {radius:.2f} "
                f"| grip[{g[0]:.2f} {g[1]:.2f} {g[2]:.2f}]")

            self.rewards_hist.append(self.ep_reward)
            self.success_hist.append(1 if success else 0)
            self.dist_hist.append(float(info["distance"]))
            self.radius_hist.append(radius)

            self.ep += 1
            self.ep_reward = 0.0
            self.ep_steps = 0

            self._maybe_tighten()          # <-- CURRICULUM

            if self.checkpoint_every and self.ep % self.checkpoint_every == 0:
                self.model.save(f"{self.tag}_ep{self.ep}")
                self.log.info(f"  [checkpoint] saved {self.tag}_ep{self.ep}.zip")

            if self.ep >= self.total_episodes:
                self.log.info(f"reached {self.total_episodes} episodes -> stopping")
                return False
        return True


def plot_curve(logger, rewards, success, dists, radii, path, title):
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

    # final distance vs the curriculum radius it must beat
    axes[2].plot(dists, alpha=0.35, color="darkred", label="final distance")
    axes[2].plot(rolling(dists), lw=2, color="red", label="rolling mean")
    axes[2].plot(radii, lw=2, ls="--", color="gray", label="approach_radius (curriculum)")
    axes[2].set_xlabel("episode"); axes[2].set_ylabel("distance (m)")
    axes[2].set_title("Final distance vs handoff radius  (SUCCESS when red dips below grey)")
    axes[2].legend()

    fig.tight_layout(); fig.savefig(path, dpi=120)
    logger.info(f"saved learning curve -> {path}")


def run_training(agent_cls, tag, total_episodes=500, max_ep_steps=500,
                 detail_every=25, checkpoint_every=50,
                 curriculum=True, curric_window=10, curric_thresh=0.70,
                 curric_step=0.10, curric_floor=0.15, **agent_kwargs):
    """Register the env, build the agent, train with logging + curriculum + graphs."""
    logger = build_logger(f"training_{tag}.log")

    register(id="HarnessPick-v0", entry_point=lambda: make_env(),
             max_episode_steps=max_ep_steps)
    env = gym.make("HarnessPick-v0")

    agent = agent_cls("MlpPolicy", env, verbose=0, **agent_kwargs)

    cb = EpisodeLogger(logger, total_episodes, detail_every, checkpoint_every, tag,
                       curriculum=curriculum, curric_window=curric_window,
                       curric_thresh=curric_thresh, curric_step=curric_step,
                       curric_floor=curric_floor)

    start_radius = float(getattr(env.unwrapped, "approach_radius", 0.0))
    logger.info(f"=== training {tag.upper()} for {total_episodes} episodes "
                f"(max {max_ep_steps} steps/ep, headless) "
                f"| start approach_radius {start_radius:.2f} "
                f"| curriculum {'ON' if curriculum else 'OFF'} ===")

    agent.learn(total_timesteps=total_episodes * max_ep_steps, callback=cb)

    agent.save(tag)
    logger.info(f"saved final policy -> {tag}.zip  (load it in watch.py to SEE it)")
    plot_curve(logger, cb.rewards_hist, cb.success_hist, cb.dist_hist, cb.radius_hist,
               f"learning_curve_{tag}.png", tag.upper())
