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

# --- paths -------------------------------------------------------------------
# Resolved relative to the folder that CONTAINS the harness_rl package, so the
# snapshot can never be silently picked up from the current working directory.
# Override with the HARNESS_XML / HARNESS_SNAPSHOT environment variables.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

XML      = os.environ.get("HARNESS_XML",      os.path.join(_ROOT, "Multi Gripper 5.xml"))
SNAPSHOT = os.environ.get("HARNESS_SNAPSHOT", os.path.join(_ROOT, "S0_harness.npy"))


def _check_assets():
    """Checked when the env is BUILT (not at import), so `import harness_rl`
    always succeeds and gym.make() gives a clear message if assets are missing."""
    if not os.path.isfile(XML):
        raise FileNotFoundError(
            f"MuJoCo model not found: {XML}\n"
            f"Put the .xml (and its .stl meshes) there, or set HARNESS_XML.")
    if not os.path.isfile(SNAPSHOT):
        raise FileNotFoundError(
            f"Harness snapshot not found: {SNAPSHOT}\n"
            f"Generate it with harness_rl.save_snapshot() at the harness-formed "
            f"moment, or set HARNESS_SNAPSHOT.")
    # A snapshot belongs to ONE model. If you change the XML you MUST re-capture it.
    n = np.load(SNAPSHOT).size
    print(f"[config] model    : {XML}")
    print(f"[config] snapshot : {SNAPSHOT}  ({n} values)")


# --- wire / gripper identification --------------------------------------------
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

# --- connector targets ---------------------------------------------------------
GA_target_pos_1 = [6.840,  0.215, 3.2]    # Bedkom
GA_target_pos_2 = [7.020,  0.125, 3.2]    # Taststart
GA_target_pos_3 = [7.020,  0.026, 3.2]    # Schaltbet_A
GA_target_pos_4 = [7.000,  0.080, 3.2]    # LSREGLER
GA_target_pos_5 = [7.285, -0.100, 3.2]    # Schaltbet_B
GA_target_pos_6 = [6.950, -0.325, 3.2]    # ANTKESSY1
GA_target_pos_7 = [6.700, -0.025, 3.2]    # LADE1_B
GA_target_pos_8 = [6.500,  0.230, 3.2]    # LADE1_A

TARGET_XYZ = tuple(GA_target_pos_2)       # TASTSTART

# --- phased grasp (TASTSTART choreography) -------------------------------------
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

        # ---- motion ----
        # substeps: physics steps per env step. At 10 the PID reaches only ~18% of the
        # commanded delta before the next action arrives and the gripper crawls; 50 gives ~75%.
        substeps=50, max_step_m=0.04,
        # SMOOTH MOTION: keep a persistent setpoint that accumulates and is ramped across the
        # substeps, so the PID never "arrives" and decelerates. Without this the gripper
        # accelerates and stops once per env step (the step-pause-step motion).
        smooth_setpoint=True,
        max_lead=None,           # default 3 * max_step_m (anti-windup on the setpoint)
        render_every=5,          # substeps between viewer syncs (run_real.py lowers this)

        # ---- task difficulty ----
        grasp_tol=0.15,
        # approach_radius = 0.0  ->  NO scripted handoff: the policy flies 100% of the path.
        # Set it to e.g. 0.80 (plus the curriculum in train.py) if the agent cannot get a
        # first success — with no handoff and no curriculum there is no partial credit.
        approach_radius=0.0,

        # ---- reward weights ----
        stress_source="passive", sigma_ref=1.0,
        w_progress=50.0, w_stress=5.0, w_coll=0.05,
        w_smooth=0.05, w_path=0.0, w_ctrl=0.0,
        w_time=1.0,
        success_bonus=200.0,

        grasp_phases=GRASP_PHASES,
        saved_state=np.load(SNAPSHOT),
        render_mode=render_mode,
    )
