"""
=============================================================================
 harness_rl/env.py  —  Gymnasium environment for multi-gripper harness picking
=============================================================================

The environment defines the PROBLEM only:
  * what the gripper senses            -> observation  (_get_obs)
  * what actions it can take           -> action_space (XYZ delta)
  * what a good outcome is             -> reward       (_compute_reward)  <-- add terms here
  * how an episode starts / ends       -> reset / step
  * the scripted grasp on success      -> _grasp_sequence

It does NOT contain an RL agent. The agent (SAC) lives in train.py and USES
this environment through the standard Gymnasium reset()/step() interface.

Section map:
  1. State spec + save_snapshot() helper
  2. HarnessPickEnv
     2a. __init__            : configuration & one-time setup
     2b. low-level helpers   : gripper pose, stress, collision
     2c. observation         : _get_obs
     2d. REWARD              : _compute_reward   <-- ADD NEW PENALTIES HERE
     2e. Gymnasium API       : reset / step
     2f. grasp sequence      : _grasp_sequence
     2g. diagnostics         : debug_signals
=============================================================================
"""

import numpy as np
import mujoco as mj
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# 1. STATE SPEC + SNAPSHOT HELPER
# =============================================================================
# Physical configuration only (time + joint pos/vel + equality-constraint
# activation). Excludes actuator state so the snapshot is independent of
# motor-vs-PID actuators; the PID integrator resets fresh each episode.
S0_SPEC = int(mj.mjtState.mjSTATE_TIME | mj.mjtState.mjSTATE_QPOS |
              mj.mjtState.mjSTATE_QVEL | mj.mjtState.mjSTATE_EQ_ACTIVE)


def save_snapshot(model, data, path="S0_harness.npy"):
    """Capture the RL reset state using the SAME spec the env restores with.
    Call on the LIVE data at the harness-formed moment:
        from harness_rl import save_snapshot
        save_snapshot(self.model, self.data, "S0_harness.npy")
    """
    s = np.zeros(mj.mj_stateSize(model, S0_SPEC))
    mj.mj_getState(model, data, s, S0_SPEC)
    np.save(path, s)
    print(f"[save_snapshot] wrote {path}  size={s.size}")
    return s


# =============================================================================
# 2. THE ENVIRONMENT
# =============================================================================
class HarnessPickEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    # -------------------------------------------------------------------------
    # 2a. __init__  — configuration & one-time setup
    # -------------------------------------------------------------------------
    def __init__(
        self,
        xml_path,
        target_xyz=None, target_site=None,
        gripper_body="Gripping_body",
        ga_actuators=("GA_X", "GA_Y", "GA_Z"),
        # wire (cable) identification
        wire_joint_prefixes=("Wire_",), wire_geom_prefix="Wire_", wire_geom_group=None,
        gripper_body_names=None,
        # workspace / dynamics
        workspace_lo=(6.5, -0.5, 2.9), workspace_hi=(8.0, 1.8, 3.7),
        substeps=10, max_step_m=0.03, grasp_tol=0.03,
        approach_radius=0.0,   # within this distance, a scripted P-controller finishes the approach
                               #   (0 = pure RL). Set e.g. 0.3 so RL only has to get CLOSE.
        constant_step=False,   # True -> every step commands the SAME delta magnitude (max_step_m),
                               #   using only the action's DIRECTION. (Actual travel still varies
                               #   with PID dynamics; usually you do NOT want this for RL.)
        # ---- SMOOTH MOTION ----
        smooth_setpoint=True,  # True  -> keep a PERSISTENT setpoint that accumulates and is
                               #          RAMPED across the substeps, so the PID never "arrives"
                               #          and decelerates -> continuous motion.
                               # False -> old behaviour: setpoint = current pos + delta, which
                               #          makes the gripper accelerate/stop once per env step.
        max_lead=None,         # how far the setpoint may run ahead of the actual gripper
                               #   (anti-windup). Default = 3 * max_step_m.
        render_every=5,        # call substep_callback every N physics substeps (for smooth video)
        gravity_comp=True, stress_source="both", sigma_ref=300.0,
        # ---- REWARD WEIGHTS (tune here; new terms are off by default) ----
        w_progress=10.0,      # + move toward the target connector
        w_stress=1.0,         # - wire bending stress
        w_coll=0.05,          # - wire/gripper contact force
        w_smooth=0.0,         # - action change between steps (jerk)  [NEW]
        w_path=0.0,           # - distance moved per step (path length) [NEW]
        w_ctrl=0.0,           # - action magnitude (effort)             [NEW]
        w_time=1.0,           # - CONSTANT per-step cost. Without this, an agent can farm
                              #   reward by standing still (all penalties ~0). Keep it >= the
                              #   sum of the other per-step penalties. (supervisor uses -1/step)
        success_bonus=50.0,   # + one-off bonus for reaching the connector
        # grasp (phased, fired on success)
        grasp_phases=(),
        saved_state=None, render_mode=None,
    ):
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.render_mode = render_mode

        # gripper + actuators
        self.gripper_bid = self.model.body(gripper_body).id
        self.ga_ref = np.array(self.model.body(gripper_body).pos, dtype=float)
        self.act_ids = np.array([self.model.actuator(n).id for n in ga_actuators])

        # target (fixed xyz OR a live site)
        self.target_site = target_site
        self.target = None if target_site is not None else np.asarray(target_xyz, float)

        # workspace / dynamics
        self.lo, self.hi = np.asarray(workspace_lo, float), np.asarray(workspace_hi, float)
        self.substeps, self.max_step_m = substeps, max_step_m
        self.grasp_tol, self.sigma_ref = grasp_tol, sigma_ref
        self.approach_radius = approach_radius
        self.constant_step = constant_step
        self.smooth_setpoint = smooth_setpoint
        self.max_lead = 3.0 * max_step_m if max_lead is None else float(max_lead)
        self.render_every = max(1, int(render_every))
        self.substep_callback = None   # set to viewer.sync for smooth live rendering
        self._setpoint = None          # persistent PID setpoint (set in reset)
        self.stress_source = stress_source

        # reward weights
        self.w_progress, self.w_stress, self.w_coll = w_progress, w_stress, w_coll
        self.w_smooth, self.w_path, self.w_ctrl = w_smooth, w_path, w_ctrl
        self.w_time = w_time
        self.success_bonus = success_bonus

        # gravity compensation
        self.gravity_comp = gravity_comp
        self.gripper_weight = float(self.model.body_subtreemass[self.gripper_bid]
                                    * abs(self.model.opt.gravity[2]))

        # --- wire joints -> DOF indices (qfrc arrays are indexed by DOF, not joint id) ---
        self.wire_joint_ids = [j for j in range(self.model.njnt)
                               if (self.model.joint(j).name or "").startswith(tuple(wire_joint_prefixes))]
        _ndof = {int(mj.mjtJoint.mjJNT_FREE): 6, int(mj.mjtJoint.mjJNT_BALL): 3,
                 int(mj.mjtJoint.mjJNT_SLIDE): 1, int(mj.mjtJoint.mjJNT_HINGE): 1}
        self.wire_dofs = []
        for j in self.wire_joint_ids:
            adr = int(self.model.jnt_dofadr[j])
            self.wire_dofs.extend(range(adr, adr + _ndof[int(self.model.jnt_type[j])]))

        # --- geom sets for collision ---
        if wire_geom_prefix is not None:
            self.wire_geoms = {g for g in range(self.model.ngeom)
                               if (self.model.geom(g).name or "").startswith(wire_geom_prefix)}
        else:
            self.wire_geoms = {g for g in range(self.model.ngeom)
                               if self.model.geom_group[g] == wire_geom_group}
        if gripper_body_names is not None:
            gids = {self.model.body(n).id for n in gripper_body_names}
            self.gripper_geoms = {g for g in range(self.model.ngeom)
                                  if self.model.geom_bodyid[g] in gids}
        else:
            self.gripper_geoms = set(self._subtree_geoms(self.gripper_bid))

        # --- grasp phases: pre-resolve names -> ids ---
        self.grasp_phases = []
        for ph in grasp_phases:
            self.grasp_phases.append({
                "eq_on":  [self.model.equality(n).id for n in ph.get("eq_on", ())],
                "eq_off": [self.model.equality(n).id for n in ph.get("eq_off", ())],
                "ctrl":   {self.model.actuator(a).id: float(v) for a, v in ph.get("ctrl", {}).items()},
                "steps":  int(ph.get("steps", 5)),
            })
        self.render_callback = None   # set to viewer.sync to draw each grasp frame

        # --- spaces ---
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # obs = gripper xyz (3) + vector-to-target (3) + normalized stress (1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)

        # --- S0 snapshot (harness-formed reset state) ---
        self._n = mj.mj_stateSize(self.model, S0_SPEC)
        if saved_state is not None:
            self._S0 = np.asarray(saved_state, float).copy()
            assert self._S0.size == self._n, (
                f"saved_state size {self._S0.size} != expected {self._n}. "
                f"Re-capture with save_snapshot() so the spec matches.")
        else:
            mj.mj_forward(self.model, self.data)
            for _ in range(200):
                mj.mj_step(self.model, self.data)
            self._S0 = np.zeros(self._n)
            mj.mj_getState(self.model, self.data, self._S0, S0_SPEC)

        # per-episode bookkeeping (set in reset)
        self._prev_dist = None
        self._prev_action = np.zeros(3)
        self._prev_gpos = np.zeros(3)

    # -------------------------------------------------------------------------
    # 2b. low-level helpers
    # -------------------------------------------------------------------------
    def _subtree_geoms(self, body_id):
        desc = {body_id}
        for b in range(self.model.nbody):
            p = b
            while p != 0:
                p = self.model.body_parentid[p]
                if p == body_id:
                    desc.add(b); break
        return [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in desc]

    def _gripper_xyz(self):
        return np.array(self.data.body(self.gripper_bid).xpos)

    def _target_xyz(self):
        if self.target_site is not None:
            return np.array(self.data.site(self.target_site).xpos)
        return self.target

    def _stress(self):
        dofs = self.wire_dofs
        if self.stress_source == "constraint":
            return float(sum(self.data.qfrc_constraint[a] ** 2 for a in dofs))
        if self.stress_source == "passive":
            return float(sum(self.data.qfrc_passive[a] ** 2 for a in dofs))
        return float(sum((self.data.qfrc_constraint[a] ** 2 + self.data.qfrc_passive[a] ** 2)
                         for a in dofs))

    def _collision_cost(self):
        total, f = 0.0, np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if (pair & self.wire_geoms) and (pair & self.gripper_geoms):
                mj.mj_contactForce(self.model, self.data, i, f)
                total += float(np.linalg.norm(f[:3]))
        return total

    # -------------------------------------------------------------------------
    # 2c. observation
    # -------------------------------------------------------------------------
    def _get_obs(self):
        g = self._gripper_xyz()
        return np.concatenate([g, self._target_xyz() - g,
                               [self._stress() / (self.sigma_ref + 1e-9)]]).astype(np.float64)

    # -------------------------------------------------------------------------
    # 2d. REWARD  <-- ADD / EDIT REWARD TERMS HERE
    #     Each term is one labeled line. Positive = reward, negative = penalty.
    #     To add a new penalty: compute it, subtract it, and add it to `terms`.
    # -------------------------------------------------------------------------
    def _compute_reward(self, dist, action):
        g = self._gripper_xyz()

        r_progress  =  self.w_progress * (self._prev_dist - dist)                       # move toward target
        p_stress    =  self.w_stress   * min(self._stress() / self.sigma_ref, 10.0)     # wire bending
        p_collision =  self.w_coll     * min(self._collision_cost(), 10.0)              # wire contact force
        p_smooth    =  self.w_smooth   * float(np.linalg.norm(action - self._prev_action))  # jerk / smoothness
        p_path      =  self.w_path     * float(np.linalg.norm(g - self._prev_gpos))     # path length
        p_ctrl      =  self.w_ctrl     * float(np.square(action).sum())                 # control effort
        p_time      =  self.w_time                                                     # constant cost per step
        # e.g. add your own:  p_myterm = self.w_myterm * <your quantity>

        reward = r_progress - p_stress - p_collision - p_smooth - p_path - p_ctrl - p_time

        terms = {"progress": r_progress, "stress": p_stress, "collision": p_collision,
                 "smooth": p_smooth, "path": p_path, "ctrl": p_ctrl, "time": p_time}

        # bookkeeping for next step's progress / smoothness / path terms
        self._prev_dist = dist          # <-- ESSENTIAL: progress must be a PER-STEP delta.
                                        #     If this is not updated, r_progress pays out the
                                        #     cumulative (initial - current) distance every step.
        self._prev_action = np.array(action, dtype=float)
        self._prev_gpos = g
        return reward, terms

    # -------------------------------------------------------------------------
    # 2e. Gymnasium API : reset / step
    # -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_setState(self.model, self.data, self._S0, S0_SPEC)   # restores eq_active too
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        mj.mj_forward(self.model, self.data)                       # update xpos before reading it
        self.data.ctrl[self.act_ids] = self._gripper_xyz() - self.ga_ref

        self._prev_dist = float(np.linalg.norm(self._gripper_xyz() - self._target_xyz()))
        self._prev_action = np.zeros(3)
        self._prev_gpos = self._gripper_xyz()
        self._setpoint = self._gripper_xyz().copy()   # persistent PID setpoint starts here
        return self._get_obs(), {}

    def step(self, action):
        # 1) turn the action into a PID setpoint, then run physics
        g = self._gripper_xyz()
        if self._setpoint is None:
            self._setpoint = g.copy()

        # --- scripted last-mile handoff: a controller finishes the approach ---
        if self.approach_radius > 0.0:
            d = np.linalg.norm(self._target_xyz() - g)
            if d < self.approach_radius:
                ref = self._setpoint if self.smooth_setpoint else g
                action = np.clip((self._target_xyz() - ref) / self.max_step_m, -1, 1)

        if self.constant_step:
            n = float(np.linalg.norm(action))
            if n > 1e-6:
                action = np.asarray(action, float) / n

        prev_sp = self._setpoint.copy()
        if self.smooth_setpoint:
            # PERSISTENT setpoint: it keeps moving ahead, so the PID never "arrives"
            # and decelerates -> continuous motion instead of step-stop-step-stop.
            sp = self._setpoint + np.clip(action, -1, 1) * self.max_step_m
            lead = sp - g                                   # anti-windup: cap how far ahead
            L = float(np.linalg.norm(lead))
            if L > self.max_lead:
                sp = g + lead / L * self.max_lead
            self._setpoint = np.clip(sp, self.lo, self.hi)
        else:
            self._setpoint = np.clip(g + np.clip(action, -1, 1) * self.max_step_m,
                                     self.lo, self.hi)

        for s in range(self.substeps):
            # RAMP the setpoint across the substeps so the PID sees a smoothly moving
            # target rather than a jump once per env step.
            if self.smooth_setpoint:
                f = (s + 1) / self.substeps
                cur = prev_sp + (self._setpoint - prev_sp) * f
            else:
                cur = self._setpoint
            self.data.ctrl[self.act_ids] = cur - self.ga_ref
            if self.gravity_comp:
                self.data.xfrc_applied[self.gripper_bid, 2] = self.gripper_weight
            mj.mj_step(self.model, self.data)
            # render DURING the substeps -> smooth video (one frame per env step is ~10 FPS)
            if self.substep_callback is not None and (s % self.render_every == 0):
                self.substep_callback()

        # 2) reward
        dist = float(np.linalg.norm(self._gripper_xyz() - self._target_xyz()))
        reward, terms = self._compute_reward(dist, action)

        # 3) termination + grasp on success
        terminated = dist < self.grasp_tol
        if terminated:
            reward += self.success_bonus
            self._grasp_sequence()
        truncated = False   # TimeLimit (max_episode_steps at registration) sets this

        info = {"distance": dist, "is_success": terminated,
                "grip": self._gripper_xyz().tolist(),      # current gripper position (for logging)
                "target": self._target_xyz().tolist(),     # target connector position
                **terms}
        return self._get_obs(), reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # 2f. grasp sequence (phased, fired on success)
    # -------------------------------------------------------------------------
    def _grasp_sequence(self):
        hold = self._gripper_xyz() - self.ga_ref
        for ph in self.grasp_phases:
            for eid in ph["eq_on"]:
                self.data.eq_active[eid] = 1
            for eid in ph["eq_off"]:
                self.data.eq_active[eid] = 0
            for aid, val in ph["ctrl"].items():
                self.data.ctrl[aid] = val
            for _ in range(ph["steps"]):
                self.data.ctrl[self.act_ids] = hold
                if self.gravity_comp:
                    self.data.xfrc_applied[self.gripper_bid, 2] = self.gripper_weight
                mj.mj_step(self.model, self.data)
                if self.render_callback is not None:
                    self.render_callback()

    # -------------------------------------------------------------------------
    # 2g. diagnostics
    # -------------------------------------------------------------------------
    def debug_signals(self):
        dofs = self.wire_dofs
        sc = float(sum(self.data.qfrc_constraint[a] ** 2 for a in dofs))
        sp = float(sum(self.data.qfrc_passive[a] ** 2 for a in dofs))
        n, ff, f = 0, 0.0, np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if (pair & self.wire_geoms) and (pair & self.gripper_geoms):
                n += 1
                mj.mj_contactForce(self.model, self.data, i, f)
                ff += float(np.linalg.norm(f[:3]))
        return {"stress_constraint": sc, "stress_passive": sp,
                "n_contacts": n, "contact_force": ff}
