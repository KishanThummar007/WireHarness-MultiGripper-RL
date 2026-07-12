"""
=============================================================================
 run_real.py  —  WATCH the environment (this does NOT train)
=============================================================================
Runs a policy in the viewer so you can see the gripper reach + grasp.
By default it uses a simple scripted policy (go toward the target) to test
the env. To watch a TRAINED policy instead, set POLICY = "harness_sac".
Close the viewer window to stop.
=============================================================================
"""

import time
import numpy as np
import mujoco, mujoco.viewer
from gymnasium.utils.env_checker import check_env

from harness_rl import make_env

POLICY = None            # None = scripted test policy; or "harness_sac" to load a trained agent

env = make_env()
print("wire joints:", len(env.wire_joint_ids),
      "| wire geoms:", len(env.wire_geoms),
      "| gripper geoms:", len(env.gripper_geoms))
check_env(env)
print("check_env passed.\n")

# optional trained agent
agent = None
if POLICY:
    from stable_baselines3 import SAC
    agent = SAC.load(POLICY)
    print(f"loaded trained policy: {POLICY}")


def choose_action(obs):
    if agent is not None:
        a, _ = agent.predict(obs, deterministic=True)   # the trained policy
        return a
    return np.clip(obs[3:6] * 20.0, -1, 1)               # scripted: head toward target


obs, _ = env.reset(seed=0)
step = 0
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    env.render_callback = lambda: (viewer.sync(), time.sleep(0.01))   # draw each grasp frame
    prev_g = env._gripper_xyz()
    while viewer.is_running() and step < 6000:
        action = choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync(); time.sleep(0.01)
        step += 1

        g = env._gripper_xyz()
        travel = float(np.linalg.norm(g - prev_g))                 # ACTUAL distance moved this step
        cmd = float(np.linalg.norm(np.clip(action, -1, 1)) * env.max_step_m)  # commanded delta
        prev_g = g
        print(f"s{step:4d} grip[{g[0]:.2f} {g[1]:.2f} {g[2]:.2f}] dist {info['distance']:.3f} "
              f"| act[{action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f}] "
              f"| commanded {cmd:.4f} | ACTUAL travel {travel:.4f} "
              f"| rew {reward:+6.3f}")

        if terminated or truncated:
            if terminated:
                time.sleep(1.0)          # hold on the grasped connector
            obs, _ = env.reset()
            prev_g = env._gripper_xyz()

print("viewer closed, exiting.")
