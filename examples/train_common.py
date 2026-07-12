"""
=============================================================================
 examples/train_common.py  —  shared training machinery
=============================================================================
Both train_sac.py and train_td3.py import this, so BOTH get identical
episode logging, log file, learning-curve graphs, the CURRICULUM, and
RESUMABLE TRAINING.
=============================================================================
"""

import json
import os
import logging
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback

from harness_rl import make_env


# =============================================================================
# logging
# =============================================================================
def build_logger(log_file, append=False):
    logger = logging.getLogger("harness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    # append mode when resuming, so the earlier history is not wiped
    fh = logging.FileHandler(log_file, mode="a" if append else "w")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


# =============================================================================
# checkpoint helpers
# =============================================================================
def _paths(tag):
    return f"{tag}.zip", f"{tag}_buffer.pkl", f"{tag}_state.json"


def save_checkpoint(agent, cb, tag, logger):
    """Save policy + replay buffer + training state (so a later run can resume)."""
    model_p, buf_p, state_p = _paths(tag)
    agent.save(tag)                                     # policy
    try:
        agent.save_replay_buffer(buf_p)                 # experience  <-- the important one
    except Exception as e:                              # on-policy algos (PPO) have no buffer
        logger.info(f"  (no replay buffer to save: {type(e).__name__})")
    state = {
        "episodes_done": cb.ep,
        "approach_radius": float(getattr(cb._env(), "approach_radius", 0.0)),
        "rewards_hist": cb.rewards_hist,
        "success_hist": cb.success_hist,
        "dist_hist": cb.dist_hist,
        "radius_hist": cb.radius_hist,
    }
    with open(state_p, "w") as f:
        json.dump(state, f)
    logger.info(f"  [checkpoint] saved {model_p} + {buf_p} + {state_p} "
                f"(episode {cb.ep})")


def load_checkpoint(agent_cls, tag, env, logger, **agent_kwargs):
    """Return (agent, state) resuming from disk if a checkpoint exists, else a fresh agent."""
    model_p, buf_p, state_p = _paths(tag)

    if not os.path.isfile(model_p):
        logger.info("no checkpoint found -> starting a FRESH agent")
        return agent_cls("MlpPolicy", env, verbose=0, **agent_kwargs), None

    logger.info(f"found {model_p} -> RESUMING training")
    agent = agent_cls.load(model_p, env=env, **agent_kwargs)

    if os.path.isfile(buf_p):
        try:
            agent.load_replay_buffer(buf_p)
            n = agent.replay_buffer.size()
            logger.info(f"  restored replay buffer: {n} transitions "
                        f"(this is what preserves past experience)")
        except Exception as e:
            logger.info(f"  WARNING: could not load replay buffer ({e}) "
                        f"-> the agent keeps its weights but loses its memory")
    else:
        logger.info("  WARNING: no replay buffer file -> agent keeps weights but loses memory")

    state = None
    if os.path.isfile(state_p):
        with open(state_p) as f:
            state = json.load(f)
        logger.info(f"  restored state: {state['episodes_done']} episodes done, "
                    f"approach_radius {state['approach_radius']:.2f}")
    return agent, state


# =============================================================================
# episode logger + curriculum
# =============================================================================
class EpisodeLogger(BaseCallback):
    def __init__(self, logger, total_episodes, detail_every, checkpoint_every, tag,
                 curriculum=True, curric_window=10, curric_thresh=0.70,
                 curric_step=0.10, curric_floor=0.15, resume_state=None):
        super().__init__()
        self.log = logger
        self.total_episodes = total_episodes      # ADDITIONAL episodes to run this session
        self.detail_every = detail_every
        self.checkpoint_every = checkpoint_every
        self.tag = tag
        self.curriculum = curriculum
        self.curric_window = curric_window
        self.curric_thresh = curric_thresh
        self.curric_step = curric_step
        self.curric_floor = curric_floor

        self.ep_reward = 0.0
        self.ep_steps = 0

        # --- restore history so episode numbers and plots CONTINUE ---
        if resume_state:
            self.ep = int(resume_state["episodes_done"])
            self.rewards_hist = list(resume_state["rewards_hist"])
            self.success_hist = list(resume_state["success_hist"])
            self.dist_hist = list(resume_state["dist_hist"])
            self.radius_hist = list(resume_state["radius_hist"])
        else:
            self.ep = 0
            self.rewards_hist, self.success_hist = [], []
            self.dist_hist, self.radius_hist = [], []
        self.start_ep = self.ep                   # where THIS session began

    def _env(self):
        return self.training_env.envs[0].unwrapped

    def _maybe_tighten(self):
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

            self._maybe_tighten()

            # periodic checkpoint -> safe to Ctrl+C / crash / power cut at any time
            if self.checkpoint_every and (self.ep - self.start_ep) % self.checkpoint_every == 0:
                save_checkpoint(self.model, self, self.tag, self.log)

            if (self.ep - self.start_ep) >= self.total_episodes:
                self.log.info(f"ran {self.total_episodes} episodes this session "
                              f"(total {self.ep}) -> stopping")
                return False
        return True


# =============================================================================
# plotting
# =============================================================================
def plot_curve(logger, rewards, success, dists, radii, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rolling(x, k=10):
        x = np.asarray(x, float)
        return x if len(x) < k else np.convolve(x, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    axes[0].plot(rewards, alpha=0.35, label="per episode")
    axes[0].plot(rolling(rewards), lw=2, label="rolling mean (10)")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("total reward")
    axes[0].set_title(f"{title} — learning curve (all sessions)"); axes[0].legend()

    axes[1].plot(rolling(success, 10) * 100, lw=2, color="green")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("success rate (%)")
    axes[1].set_title("Rolling success rate (10 ep)"); axes[1].set_ylim(-5, 105)

    axes[2].plot(dists, alpha=0.35, color="darkred", label="final distance")
    axes[2].plot(rolling(dists), lw=2, color="red", label="rolling mean")
    axes[2].plot(radii, lw=2, ls="--", color="gray", label="approach_radius (curriculum)")
    axes[2].set_xlabel("episode"); axes[2].set_ylabel("distance (m)")
    axes[2].set_title("Final distance vs handoff radius  (SUCCESS when red dips below grey)")
    axes[2].legend()

    fig.tight_layout(); fig.savefig(path, dpi=120)
    logger.info(f"saved learning curve -> {path}")


# =============================================================================
# main entry point
# =============================================================================
def run_training(agent_cls, tag, total_episodes=500, max_ep_steps=500,
                 detail_every=25, checkpoint_every=10, resume=True,
                 curriculum=True, curric_window=10, curric_thresh=0.70,
                 curric_step=0.10, curric_floor=0.15, **agent_kwargs):
    """Train (or RESUME training) with logging, curriculum, checkpoints, graphs.

    total_episodes = how many ADDITIONAL episodes to run in THIS session.
    resume=True    = continue from <tag>.zip + <tag>_buffer.pkl if they exist.
    """
    resuming = resume and os.path.isfile(f"{tag}.zip")
    logger = build_logger(f"training_{tag}.log", append=resuming)

    register(id="HarnessPick-v0", entry_point=lambda: make_env(),
             max_episode_steps=max_ep_steps)
    env = gym.make("HarnessPick-v0")

    if resume:
        agent, state = load_checkpoint(agent_cls, tag, env, logger, **agent_kwargs)
    else:
        logger.info("resume=False -> starting a FRESH agent")
        agent, state = agent_cls("MlpPolicy", env, verbose=0, **agent_kwargs), None

    # restore the curriculum difficulty reached in the previous session
    if state is not None:
        env.unwrapped.approach_radius = float(state["approach_radius"])

    cb = EpisodeLogger(logger, total_episodes, detail_every, checkpoint_every, tag,
                       curriculum=curriculum, curric_window=curric_window,
                       curric_thresh=curric_thresh, curric_step=curric_step,
                       curric_floor=curric_floor, resume_state=state)

    radius = float(getattr(env.unwrapped, "approach_radius", 0.0))
    logger.info(f"=== {'RESUMING' if state else 'STARTING'} {tag.upper()} "
                f"| +{total_episodes} episodes this session "
                f"(already done: {cb.ep}) | max {max_ep_steps} steps/ep "
                f"| approach_radius {radius:.2f} "
                f"| curriculum {'ON' if curriculum else 'OFF'} | headless ===")

    # reset_num_timesteps=False keeps the global step counter continuous across sessions
    agent.learn(total_timesteps=total_episodes * max_ep_steps,
                callback=cb, reset_num_timesteps=not resuming)

    save_checkpoint(agent, cb, tag, logger)
    logger.info(f"session complete -> {tag}.zip  (run again to CONTINUE from here)")
    plot_curve(logger, cb.rewards_hist, cb.success_hist, cb.dist_hist, cb.radius_hist,
               f"learning_curve_{tag}.png", tag.upper())
