"""
=============================================================================
 harness_rl/config.py  —  all model-specific settings in ONE place
=============================================================================
Both run_real.py (visualize) and train.py (learn) call make_env(), so the
environment is configured identically everywhere. Edit values here only.
=============================================================================
"""

import numpy as np
from .env import HarnessPickEnv

# --- paths ---
XML = r"E:\MultiGripper_RL\Multi Gripper 4.xml"
SNAPSHOT = "S0_harness.npy"

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
    return HarnessPickEnv(
        XML,
        target_xyz=TARGET_XYZ,
        wire_joint_prefixes=WIRE_PREFIXES,
        wire_geom_prefix="Wire_",
        gripper_body_names=GRIPPER_BODY_NAMES,
        workspace_lo=(6.5, -0.5, 2.9), workspace_hi=(8.0, 1.8, 3.7),
        substeps=10, max_step_m=0.03, grasp_tol=0.03,
        # ---- reward weights ----
        stress_source="passive", sigma_ref=1.0,
        w_progress=10.0, w_stress=5.0, w_coll=0.05,
        w_smooth=0.0, w_path=0.0, w_ctrl=0.0,   # <-- raise these to add smoothness / path-length / effort penalties
        success_bonus=50.0,
        grasp_phases=GRASP_PHASES,
        saved_state=np.load(SNAPSHOT),
        render_mode=render_mode,
    )
