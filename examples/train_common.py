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
import logging
import os

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from harness_rl import make_env

# ---------------- settings ----------------
TAG              = "harness_sac"
TOTAL_EPISODES   = 20      # ADDITIONAL episodes to run this session
DETAIL_EVERY     = 10      # print per-STEP detail every N episodes (0 = never)
CHECKPOINT_EVERY = 1       # save policy + buffer + state every N episodes
MAX_EP_STEPS     = 500     # episode length cap (300 cut off near-successes)
RESUME           = True    # continue from a previous run if a checkpoint exists
LOG_FILE         = "training.log"
PLOT_FILE        = "learning_curve.png"

# ---------------- curriculum ----------------
CURRICULUM    = False      # OFF: the agent flies 100% of the path (config.py sets
                           #      approach_radius=0.0). Set True + approach_radius=0.80 if
                           #      you cannot get a first success.
CURRIC_WINDOW = 10         # look at the last N episodes
CURRIC_THRESH = 0.70       # if >= 70% of them succeeded ...
CURRIC_STEP   = 0.10       # ... tighten the radius by this much
CURRIC_FLOOR  = 0.15       # never go below this

MODEL_P, BUF_P, STATE_P = f"{TAG}.zip", f"{TAG}_buffer.pkl", f"{TAG}_state.json"
_RESUMING = RESUME and os.path.isfile(MODEL_P)

# ---------------- logging to console + file ----------------
logger = logging.getLogger("harness")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
_fh = logging.FileHandler(LOG_FILE, mode="a" if _RESUMING else "w")   # append when resuming
_fh.setFormatter(_fmt); logger.addHandler(_fh)
_sh = logging.StreamHandler(); _sh.setFormatter(_fmt); logger.addHandler(_sh)


class EpisodeLogger(BaseCallback):
    """Logs each episode, runs the CURRICULUM, checkpoints, stops after
    TOTAL_EPISODES, and records data for the learning curves."""

    def __init__(self, total_episodes, detail_every, checkpoint_every, resume_state=None):
        super().__init__()
        self.total_episodes = total_episodes    # additional episodes THIS session
        self.detail_every = detail_every
        self.checkpoint_every = checkpoint_every

        self.ep_reward = 0.0
        self.ep_steps = 0
        self._since_tighten = 0                 # episodes since the last curriculum step

        # restore history so episode numbers and plots CONTINUE across sessions
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
        self.start_ep = self.ep                 # where THIS session began

    # ---- the environment instance the agent is training on ----
    def _env(self):
        return self.training_env.envs[0].unwrapped

    def _save(self):
        """Save policy + replay buffer + state, so a later run can resume."""
        self.model.save(TAG)
        try:
            self.model.save_replay_buffer(BUF_P)
        except Exception as e:
            logger.info(f"  (no replay buffer to save: {type(e).__name__})")
        state = {
            "episodes_done": self.ep,
            "approach_radius": float(getattr(self._env(), "approach_radius", 0.0)),
            "rewards_hist": self.rewards_hist,
            "success_hist": self.success_hist,
            "dist_hist": self.dist_hist,
            "radius_hist": self.radius_hist,
        }
        with open(STATE_P, "w") as f:
            json.dump(state, f)
        logger.info(f"  [checkpoint] saved {MODEL_P} + {BUF_P} + {STATE_P} (episode {self.ep})")

    def _maybe_tighten(self):
        """If the agent succeeds reliably, shrink the scripted-handoff radius."""
        if not CURRICULUM or self._since_tighten < CURRIC_WINDOW:
            return
        recent = float(np.mean(self.success_hist[-CURRIC_WINDOW:]))
        if recent < CURRIC_THRESH:
            return
        env = self._env()
        cur = float(getattr(env, "approach_radius", 0.0))
        new = max(CURRIC_FLOOR, cur - CURRIC_STEP)
        if new < cur - 1e-9:
            env.approach_radius = new
            logger.info(f"  [curriculum] success {recent*100:.0f}% >= {CURRIC_THRESH*100:.0f}%"
                        f"  ->  approach_radius {cur:.2f} -> {new:.2f}  (task just got harder)")
            self._since_tighten = 0     # reset the WINDOW only — never falsify success_hist

    def _on_step(self) -> bool:
        action = self.locals["actions"][0]
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        self.ep_reward += reward
        self.ep_steps += 1

        if self.detail_every and (self.ep % self.detail_every == 0):
            g, t = info["grip"], info["target"]
            logger.info(
                f"  ep{self.ep:04d} s{self.ep_steps:4d} "
                f"act[{action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f}] "
                f"grip[{g[0]:.2f} {g[1]:.2f} {g[2]:.2f}] "
                f"tgt[{t[0]:.2f} {t[1]:.2f} {t[2]:.2f}] "
                f"dist {info['distance']:.3f} rew {reward:+.3f}")

        if done:
            success = bool(info.get("is_success", False))
            g = info["grip"]
            radius = float(getattr(self._env(), "approach_radius", 0.0))
            logger.info(
                f"EPISODE {self.ep:04d} | steps {self.ep_steps:4d} "
                f"| reward {self.ep_reward:+8.2f} | {'PASS' if success else 'FAIL'} "
                f"| final dist {info['distance']:.3f} | radius {radius:.2f} "
                f"| grip[{g[0]:.2f} {g[1]:.2f} {g[2]:.2f}]")

            self.rewards_hist.append(self.ep_reward)
            self.success_hist.append(1 if success else 0)
            self.dist_hist.append(float(info["distance"]))
            self.radius_hist.append(radius)

            self.ep += 1
            self._since_tighten += 1        # <-- ESSENTIAL: without this the curriculum
                                            #     fires once and then never again
            self.ep_reward = 0.0
            self.ep_steps = 0

            self._maybe_tighten()           # curriculum

            if self.checkpoint_every and (self.ep - self.start_ep) % self.checkpoint_every == 0:
                self._save()

            if (self.ep - self.start_ep) >= self.total_episodes:
                logger.info(f"ran {self.total_episodes} episodes this session "
                            f"(total {self.ep}) -> stopping")
                return False
        return True


def plot_curve(rewards, success, dists, radii, path):
    import matplotlib
    matplotlib.use("Agg")            # headless backend (no display needed)
    import matplotlib.pyplot as plt

    def rolling(x, k=10):
        x = np.asarray(x, float)
        if len(x) < k:
            return x
        return np.convolve(x, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    axes[0].plot(rewards, alpha=0.35, label="per episode")
    axes[0].plot(rolling(rewards), lw=2, label="rolling mean (10)")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("total reward")
    axes[0].set_title("Learning curve (all sessions)"); axes[0].legend()

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


def main():
    register(id="HarnessPick-v0", entry_point=lambda: make_env(), max_episode_steps=MAX_EP_STEPS)
    env = gym.make("HarnessPick-v0")

    # ---------- build or RESUME the agent ----------
    state = None
    if _RESUMING:
        logger.info(f"found {MODEL_P} -> RESUMING training")
        agent = SAC.load(MODEL_P, env=env)
        if os.path.isfile(BUF_P):
            agent.load_replay_buffer(BUF_P)
            logger.info(f"  restored replay buffer: {agent.replay_buffer.size()} transitions "
                        f"(this is what preserves past experience)")
        else:
            logger.info("  WARNING: no replay buffer file -> weights kept, MEMORY LOST")
        if os.path.isfile(STATE_P):
            with open(STATE_P) as f:
                state = json.load(f)
            env.unwrapped.approach_radius = float(state["approach_radius"])
            logger.info(f"  restored state: {state['episodes_done']} episodes done, "
                        f"approach_radius {state['approach_radius']:.2f}")
    else:
        logger.info("no checkpoint -> starting a FRESH agent")
        agent = SAC("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                    buffer_size=200_000, batch_size=256, gamma=0.99)

    cb = EpisodeLogger(TOTAL_EPISODES, DETAIL_EVERY, CHECKPOINT_EVERY, resume_state=state)

    radius = float(getattr(env.unwrapped, "approach_radius", 0.0))
    logger.info(f"=== {'RESUMING' if state else 'STARTING'} | +{TOTAL_EPISODES} episodes "
                f"this session (already done: {cb.ep}) | max {MAX_EP_STEPS} steps/ep "
                f"| approach_radius {radius:.2f} "
                f"| curriculum {'ON' if CURRICULUM else 'OFF'} ===")

    # reset_num_timesteps=False keeps the global step counter continuous across sessions
    agent.learn(total_timesteps=TOTAL_EPISODES * MAX_EP_STEPS,
                callback=cb, reset_num_timesteps=not _RESUMING)

    cb._save()
    logger.info(f"session complete -> {MODEL_P}  (run again to CONTINUE from here)")
    plot_curve(cb.rewards_hist, cb.success_hist, cb.dist_hist, cb.radius_hist, PLOT_FILE)


if __name__ == "__main__":
    main()
