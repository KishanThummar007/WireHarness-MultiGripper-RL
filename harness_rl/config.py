"""
=============================================================================
 harness_rl/config.py  —  all model-specific settings in ONE place
=============================================================================
Both run_real.py (visualize) and train.py (learn) call make_env(), so the
environment is configured identically everywhere. Edit values here only.
=============================================================================
"""

import os
import numpy as np
from .env import HarnessPickEnv

# --- paths (PORTABLE: resolved relative to this package, so the repo works
#     on any machine and any OS after `git clone` + `pip install -e .`) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(os.path.dirname(_HERE), "assets")

XML = os.environ.get("HARNESS_XML", os.path.join(_ASSETS, "Multi Gripper 4.xml"))
SNAPSHOT = os.environ.get("HARNESS_SNAPSHOT", os.path.join(_ASSETS, "S0_harness.npy"))


def _check_assets():
    """Checked when the env is BUILT (not at import), so `import harness_rl`
    always succeeds and gym.make() gives a clear message if assets are missing."""
    if not os.path.isfile(XML):
        raise FileNotFoundError(
            f"MuJoCo model not found: {XML}\n"
            f"Place 'Multi Gripper 4.xml' and its .stl meshes in assets/, "
            f"or set the HARNESS_XML environment variable.")
    if not os.path.isfile(SNAPSHOT):
        raise FileNotFoundError(
            f"Harness snapshot not found: {SNAPSHOT}\n"
            f"Generate it with harness_rl.save_snapshot() at the harness-formed "
            f"moment, or set the HARNESS_SNAPSHOT environment variable.")

# --- wire / gripper identification (from your Sim_Main_3 / TrajOpt_3) ---
WIRE_PREFIXES = (
    "Wire_Schaltbet_A", "Wire_Schaltbet_B", "Wire_TASTSTART", "Wire_BEDKOM",
    "Wire_SPC_1", "Wire_SPC_2", "Wire_LSREGLER", "Wire_ANTKESSY1",
    "Wire_LADE1_A", "Wire_LADE1_B",
)
GRIPPER_BODY_NAMES = {
    "Gripping_body", "Sch_A_F1", "Sch_A_F2", "Sch_B_F1", "Sch_B_F2",
    "TAST_F1", "TAST_F2", "BED_F1", "BED_F2", "LSREG_F1", "LSREG_F2",
    "LADE1_A_F1", "LADE1_A_F2", "LADE1_B_F1", "LADE1_B_F2", "ANT_F1", "ANT_F2",
}

# --- which connector to reach this episode (TASTSTART) ---
TARGET_XYZ = (7.0, 0.12, 3.2)

# --- phased grasp (mirrors your scripted TASTSTART choreography) ---
GRASP_PHASES = (
    {"eq_off": ("R11C1_MB_3",),
     "ctrl": {"TAST_F_1_Actuator": -3.0, "TAST_F_2_Actuator": -3.0, "MODULE_BOX_3_Z": -2.0},
     "steps": 25},
    {"ctrl": {"TAST_F_1_Actuator": 3.0, "TAST_F_2_Actuator": 3.0, "MODULE_BOX_3_Z": 0.0},
     "steps": 20},
    {"eq_on": ("G3_MBC3_C1", "G3_MBC3_C2"), "eq_off": ("MB3_MBC1",), "steps": 8},
    {"steps": 10},
)


def make_env(render_mode=None):
    """Build the fully-configured environment for the real harness model."""
    _check_assets()
    return HarnessPickEnv(
        XML,
        target_xyz=TARGET_XYZ,
        wire_joint_prefixes=WIRE_PREFIXES,
        wire_geom_prefix="Wire_",
        gripper_body_names=GRIPPER_BODY_NAMES,
        workspace_lo=(6.5, -0.5, 2.9), workspace_hi=(8.0, 1.8, 3.7),
        # substeps: physics steps per env step. With only 10, the PID reaches just ~18% of the
        # commanded delta before the next action arrives -> the gripper crawls. 50 gives ~75%
        # tracking, so the agent's actions actually move the gripper.
        substeps=50, max_step_m=0.04,
        grasp_tol=0.15,          # LOOSER than 0.03 so the agent can actually get first successes
        # approach_radius: within this distance a scripted controller finishes the approach.
        # START GENEROUS (0.8). If it is too tight (e.g. 0.3) the agent never reaches the handoff
        # zone, never earns the success bonus, and has NO learning signal -> flat 0% success.
        # train_common.py shrinks this automatically as the success rate rises (curriculum).
        approach_radius=0.80,
        # ---- reward weights ----
        stress_source="passive", sigma_ref=1.0,
        # w_progress must be large enough that moving toward the goal OUT-EARNS the time
        # penalty. With w_progress=10 and w_time=1, even an optimal policy nets -0.66/step
        # (good behaviour punished, no gradient). With w_progress=50 it nets +0.75/step.
        w_progress=50.0, w_stress=5.0, w_coll=0.05,
        # NOTE: w_smooth/w_path/w_ctrl are PENALTIES. If they are large and w_time=0, the agent
        # farms reward by STANDING STILL (all penalties ~0). Keep them small, and always keep
        # w_time > 0 so doing nothing is strictly punished.
        w_smooth=0.05, w_path=0.0, w_ctrl=0.0,
        w_time=1.0,              # constant per-step cost -> idling loses, finishing fast wins
        success_bonus=200.0,     # large enough that a successful episode is clearly POSITIVE
                                 #   (reach ~ +180, idle ~ -300 over a 300-step episode)
        grasp_phases=GRASP_PHASES,
        saved_state=np.load(SNAPSHOT),
        render_mode=render_mode,
    )
