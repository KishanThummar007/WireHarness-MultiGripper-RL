# WireHarness-MultiGripper-RL

A Gymnasium environment for picking multi-branch wire harnesses with a robotic multi-gripper, simulated in MuJoCo with deformable cable physics.

[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0+-blue.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A [Gymnasium](https://gymnasium.farama.org/)-compatible reinforcement learning environment for
**realistic multi-branch wire-harness picking** with a multi-gripper end-effector, simulated in
[MuJoCo](https://mujoco.org/).

An agent navigates a multi-gripper through a dense, physically-simulated wire harness (10 cable
branches, 200+ equality constraints) to reach a connector housing while minimising induced wire
stress and collisions. On arrival, a scripted, phased grasp detaches the connector from its module
box and connects it to the gripper fingers.

---
<img width="3064" height="1738" alt="Wire_Multi-Gripper_Setup" src="https://github.com/user-attachments/assets/88cfd829-d581-4d5a-94b8-fc38056bbfa4" />

---

## Installation

```bash
git clone https://github.com/KishanThummar007/WireHarness-MultiGripper-RL.git
cd WireHarness-MultiGripper-RL
pip install -e .            # environment only
pip install -e ".[train]"   # + stable-baselines3 and matplotlib for the examples
```

**Dependencies:** `gymnasium >= 1.0`, `mujoco >= 3.1`, `numpy`

---

## Quick Start

```python
import gymnasium as gym
import harness_rl                      # registers the environment ID

env = gym.make("HarnessPick-v0")       # fixed target connector housing

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

---

## Environment Details

### Registered IDs

| ID | Target | Description |
| --- | --- | --- |
| `HarnessPick-v0` | Fixed (CON_T) | Reach `G_1` and grasp CON_T connector housing |
| `HarnessPick-v0` | Optional (CON_BK) | Reach `G_2` and grasp CON_BK connector housing |
| `HarnessPick-v0` | Optional (CON_B) | Reach `G_3` and grasp CON_B connector housing |
| `HarnessPick-v0` | Optional (CON_A) | Reach `G_4` and grasp CON_A connector housing |
| `HarnessPick-v0` | Optional (CON_LS) | Reach `G_5` and grasp CON_LS connector housing |
| `HarnessPick-v0` | Optional (CON_LA) | Reach `G_6` and grasp CON_LA connector housing |
| `HarnessPick-v0` | Optional (CON_LB) | Reach `G_7` and grasp CON_LB connector housing |
| `HarnessPick-v0` | Optional (CON_ANT) | Reach `G_8` and grasp CON_ANT connector housing |

### Observation Space

`Box(-inf, inf, shape=(7,), dtype=float64)`

| Indices | Content |
| --- | --- |
| 0 – 2 | Gripper position (x, y, z) in world coordinates |
| 3 – 5 | Vector from gripper to the target connector |
| 6 | Normalised wire stress |

### Action Space

`Box(-1.0, 1.0, shape=(3,), dtype=float32)` — a normalised (Δx, Δy, Δz) displacement, scaled by
`max_step_m` and applied as a setpoint to MuJoCo PID actuators driving the gripper gantry.

### Reward

| Term | Value | Default weight |
| --- | --- | --- |
| Progress toward the connector | `+w_progress × (prev_dist − dist)` | `10.0` |
| Wire stress (cable bending) | `−w_stress × normalised stress` | `5.0` |
| Wire/gripper contact force | `−w_coll × contact force` | `0.05` |
| Action change (smoothness) | `−w_smooth × ‖aₜ − aₜ₋₁‖` | `0.05` |
| Path length | `−w_path × ‖pₜ − pₜ₋₁‖` | `0.0` |
| Control effort | `−w_ctrl × ‖aₜ‖²` | `0.0` |
| **Per-step time cost** | **`−w_time`** | **`1.0`** |
| Connector reached | `+success_bonus` | `50.0` |

All terms are itemised in the `info` dict each step.

> ⚠️ **Reward-design warning — keep `w_time > 0`.**
> `w_smooth`, `w_path`, and `w_ctrl` are *penalties for moving*. If they are large while
> `w_time = 0`, the optimal policy is to **stand still**: no motion means no penalties, so the
> agent farms a positive return while never reaching the connector. The constant per-step cost `w_time` makes
> idling strictly worse than acting, which is what forces the agent to actually solve the task.

### `info` Dictionary

```python
{
    "distance": float,       # gripper-to-connector distance
    "is_success": bool,      # connector reached (episode terminated)
    "grip": [x, y, z],       # gripper position
    "target": [x, y, z],     # connector position
    "progress": float,       # ... plus every reward term, itemised:
    "stress": float, "collision": float, "smooth": float,
    "path": float, "ctrl": float, "time": float,
}
```

---

## Training

```bash
python examples/train_sac.py    # SAC
python examples/train_td3.py    # TD3  (same env, one-line algorithm swap)
```

Both are **headless** and produce identical artifacts:

| File | Contents |
| --- | --- |
| `harness_<algo>.zip` | the trained policy (load in `watch.py`) |
| `training_harness_<algo>.log` | per-episode PASS/FAIL, reward, distance; per-step detail |
| `learning_curve_harness_<algo>.png` | reward, rolling success rate, final-distance graphs |
| `harness_<algo>_ep<N>.zip` | periodic checkpoints — watch progress mid-training |

Then watch the result:

```bash
python examples/watch.py        # set POLICY = "harness_sac" at the top
```
---

## Repository Structure

```
WireHarness-MultiGripper-RL/
│
├── 📂 harness_rl/                  ← THE ENVIRONMENT (the "problem")
│   ├── 📄 __init__.py                registers the "HarnessPick-v0" env ID
│   ├── 📄 env.py                     HarnessPickEnv: observation, REWARD, reset, step, grasp
│   └── 📄 config.py                  settings: paths, target, workspace, reward weights,
│                                     grasp phases, approach_radius (curriculum start)
│
├── 📂 examples/                    ← THE AGENTS (the "solution")
│   ├── 📄 train_common.py            shared: episode logging, curriculum, checkpoint/RESUME, graphs
│   ├── 📄 train_sac.py               train with SAC   → harness_sac.zip
│   ├── 📄 train_td3.py               train with TD3   → harness_td3.zip
│   └── 📄 watch.py                   load a .zip policy and SEE it in the MuJoCo viewer
│
├── 📂 assets/                      ← THE MODEL
│   ├── 📄 Multi Gripper 4.xml        MuJoCo model (GA_X/Y/Z are mujoco.pid actuators)
│   ├── 📄 *.stl                      meshes
│   └── 📄 S0_harness.npy             formed-harness snapshot (the RL reset state)
│
├── 📄 pyproject.toml
├── 📄 LICENSE
└── 📄 README.md

Generated during training (git-ignored):
   harness_sac.zip                policy weights
   harness_sac_buffer.pkl         replay buffer — past experience (RESUME needs this)
   harness_sac_state.json         episode count, curriculum radius, histories
   training_harness_sac.log       per-episode PASS/FAIL + per-step detail
   learning_curve_harness_sac.png reward / success rate / distance-vs-radius
```

### How the pieces fit together

```
        config.py  ────────────────► settings + reward weights
                                                     │
                                                     ▼
       ┌──────────────────┐   action (Δx,Δy,Δz)  ┌──────────────────┐
       │    RL AGENT      │ ──────────────────►  │   ENVIRONMENT    │
       │   SAC  /  TD3    │                      │     env.py       │
       │   (examples/)    │ ◄──────────────────  │   (harness_rl)   │
       └──────────────────┘   obs + reward       └──────────────────┘
          │            ▲                                  ▲
          │ saves      │ resumes                          │ loads
          ▼            │                                  │
   ┌─────────────────────────────────┐                    │
   │  CHECKPOINT  (every N episodes) │                    │
   │   harness_sac.zip      policy   │                    │
   │   ..._buffer.pkl       memory   │ ← the replay buffer│is what
   │   ..._state.json       progress │   preserves learning across runs
   └─────────────────────────────────┘                    │
          │                                               │
          │ policy .zip                                   │
          └──────────────────────────────────────► watch.py  (MuJoCo viewer)

   CURRICULUM: approach_radius starts at 0.80 (a scripted controller finishes the
   approach inside it). Once ≥70% of the last 10 episodes PASS, it shrinks by 0.10
   (floor 0.15) — pushing the policy from "get roughly close" → "reach precisely".
```

**Train headless → watch locally.
** `train_*.py` opens **no** graphics window (fast, and works over
SSH on a remote server or cluster). It saves the policy as a `.zip`. To *see* the trained gripper,
run `watch.py`, which loads that `.zip` and renders it in the MuJoCo viewer.

---

## Customising

- **Reward** — `harness_rl/env.py` → `HarnessPickEnv._compute_reward()`. Each term is a single
  labelled line; add your own alongside them.
- **Weights and settings** — `harness_rl/config.py`.
- **RL algorithm** — `examples/`. Swapping SAC → TD3 → PPO is a one-line change.

### Key tuning notes

- **`substeps`** (default `50`) is the number of physics steps per env step. Too few and the PID
  controller cannot reach its setpoint before the next action arrives — at `substeps=10` the gripper
  achieves only ~18% of the commanded motion and effectively crawls. `50` gives ~75% tracking.
- **`approach_radius`** (default `0.30`) hands the final approach to a scripted P-controller, so RL
  only has to learn to get *close*. This is what makes the task learnable at all.

---


## License

MIT — see [LICENSE](LICENSE).
