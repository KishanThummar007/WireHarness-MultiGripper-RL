"""
=============================================================================
 check_repo.py  —  self-audit before publishing
=============================================================================
Run this from the ROOT of a FRESH clone:

    git clone https://github.com/KishanThummar007/WireHarness-MultiGripper-RL.git
    cd WireHarness-MultiGripper-RL
    pip install -e ".[train]"
    python check_repo.py

It checks every failure mode this project actually hit. If everything passes,
the repo is ready to submit to Farama.
=============================================================================
"""

import os
import sys
import glob
import inspect
import traceback

OK, BAD, WARN = "PASS", "FAIL", "WARN"
results = []


def check(name, fn):
    try:
        status, msg = fn()
    except Exception as e:
        status, msg = BAD, f"{type(e).__name__}: {e}"
    results.append((status, name, msg))
    icon = {OK: "[PASS]", BAD: "[FAIL]", WARN: "[WARN]"}[status]
    print(f"{icon} {name}\n        {msg}")


# --- 1. repo layout --------------------------------------------------------
def c_layout():
    need = ["pyproject.toml", "README.md", "LICENSE",
            "harness_rl/__init__.py", "harness_rl/env.py", "harness_rl/config.py"]
    missing = [p for p in need if not os.path.isfile(p)]
    return (BAD, f"missing: {missing}") if missing else (OK, "all core files present")


# --- 2. assets (XML + meshes + snapshot) -----------------------------------
def c_assets():
    xmls = glob.glob("assets/*.xml")
    stls = glob.glob("assets/*.stl") + glob.glob("assets/*.STL")
    npys = glob.glob("assets/*.npy")
    if not xmls:
        return BAD, "no .xml in assets/ -> nobody can build the model"
    if not stls:
        return BAD, "no .stl meshes in assets/ -> the model will NOT compile for anyone"
    if not npys:
        return BAD, "no S0_harness.npy in assets/ -> the env cannot reset the formed harness"
    return OK, f"{len(xmls)} xml, {len(stls)} stl, {len(npys)} npy"


# --- 3. NO hardcoded absolute paths (the release blocker) ------------------
def c_paths():
    src = open("harness_rl/config.py").read()
    bad = [tok for tok in ("E:\\", "C:\\", "/home/", "/Users/") if tok in src]
    if bad:
        return BAD, (f"hardcoded absolute path(s) {bad} in config.py -> "
                     f"the env is UNUSABLE on any other machine")
    if "os.path.dirname" not in src:
        return WARN, "config.py does not look like it resolves paths relative to the package"
    return OK, "paths are portable (resolved relative to the package)"


# --- 4. env.py: the _prev_dist bug ------------------------------------------
def c_prev_dist():
    import harness_rl.env as E
    src = inspect.getsource(E.HarnessPickEnv._compute_reward)
    if "self._prev_dist = dist" not in src:
        return BAD, ("_compute_reward does NOT update self._prev_dist -> the progress term pays "
                     "CUMULATIVE distance every step (rewards balloon into the thousands)")
    init = inspect.getsource(E.HarnessPickEnv.__init__)
    if "self._prev_dist = None" not in init:
        return WARN, "__init__ should set self._prev_dist = None (not a value)"
    return OK, "progress is a per-step delta, and __init__ is correct"


# --- 5. config.py: reward weights + curriculum start -------------------------
def c_weights():
    import harness_rl.config as C
    import re
    src = inspect.getsource(C.make_env)
    # strip comments so we test the ACTUAL values, not numbers mentioned in comments
    code = "\n".join(line.split("#")[0] for line in src.splitlines())
    problems = []
    m = re.search(r"approach_radius\s*=\s*([0-9.]+)", code)
    if m:
        r = float(m.group(1))
        if r < 0.5:
            problems.append(f"approach_radius={r} is TOO TIGHT -> agent never reaches the "
                            f"handoff zone, 0% success forever (use 0.8 + curriculum)")
    m = re.search(r"w_time\s*=\s*([0-9.]+)", code)
    if m and float(m.group(1)) <= 0:
        problems.append("w_time<=0 -> the agent can farm reward by STANDING STILL")
    m = re.search(r"w_progress\s*=\s*([0-9.]+)", code)
    if m and float(m.group(1)) < 30:
        problems.append(f"w_progress={m.group(1)} is small vs w_time -> even an optimal policy "
                        f"is punished per step (no gradient). Use ~50.")
    m = re.search(r"substeps\s*=\s*([0-9]+)", code)
    if m and int(m.group(1)) < 25:
        problems.append(f"substeps={m.group(1)} -> the PID reaches only ~18% of the commanded "
                        f"motion; the gripper crawls. Use 50.")
    return (BAD, " | ".join(problems)) if problems else (OK, "reward weights and curriculum sane")


# --- 6. the public API: import registers the env, gym.make works -------------
def c_gym():
    import gymnasium as gym
    import harness_rl                      # must register on import
    ids = [s for s in gym.registry if "HarnessPick" in s]
    if not ids:
        return BAD, "import harness_rl did NOT register an env id -> gym.make will fail"
    env = gym.make(ids[0])
    obs, info = env.reset(seed=0)
    obs, r, te, tr, info = env.step(env.action_space.sample())
    env.close()
    return OK, f"gym.make('{ids[0]}') + reset + step all work"


# --- 7. Gymnasium API compliance --------------------------------------------
def c_check_env():
    import warnings
    import gymnasium as gym
    import harness_rl
    from gymnasium.utils.env_checker import check_env
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(gym.make("HarnessPick-v0").unwrapped, skip_render_check=True)
    return OK, "passes gymnasium.utils.env_checker.check_env"


# --- 8. examples import cleanly ---------------------------------------------
def c_examples():
    need = ["examples/train_sac.py", "examples/train_td3.py", "examples/watch.py"]
    missing = [p for p in need if not os.path.isfile(p)]
    if missing:
        return WARN, f"missing example(s): {missing}"
    if os.path.isfile("examples/train_common.py"):
        src = open("examples/train_common.py").read()
        if "resume" not in src:
            return WARN, "train_common.py has no RESUME support (training restarts from scratch)"
        if "curriculum" not in src:
            return WARN, "train_common.py has no CURRICULUM (approach_radius never tightens)"
    return OK, "examples present, with curriculum + resume"


if __name__ == "__main__":
    print("=" * 72)
    print(" REPO AUDIT — run from the root of a FRESH clone")
    print("=" * 72)
    check("1. repo layout", c_layout)
    check("2. assets (xml + stl meshes + snapshot)", c_assets)
    check("3. NO hardcoded absolute paths", c_paths)
    check("4. env.py: _prev_dist per-step delta", c_prev_dist)
    check("5. config.py: reward weights + curriculum", c_weights)
    check("6. import registers env; gym.make works", c_gym)
    check("7. Gymnasium check_env compliance", c_check_env)
    check("8. examples (curriculum + resume)", c_examples)

    print("=" * 72)
    fails = [r for r in results if r[0] == BAD]
    warns = [r for r in results if r[0] == WARN]
    if fails:
        print(f" NOT READY — {len(fails)} blocking issue(s):")
        for _, name, msg in fails:
            print(f"   - {name}: {msg}")
        sys.exit(1)
    print(f" READY TO PUBLISH  ({len(warns)} warning(s))")
    for _, name, msg in warns:
        print(f"   ! {name}: {msg}")
