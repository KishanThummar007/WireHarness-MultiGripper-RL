"""
=============================================================================
 check_snapshot.py  —  does S0_harness.npy belong to the XML you are using?
=============================================================================
A snapshot stores the physics state of ONE specific model. Change the XML
(e.g. more wire segments) and the state size changes, so an old snapshot can
no longer be loaded.

Run:   python check_snapshot.py
It prints the size the CONFIGURED model needs, the size the CONFIGURED
snapshot actually has, and whether they match.

If they do not match, re-capture the snapshot FROM THE SAME XML:
  1. open your prestage script (Sim_Main...) and make sure it loads the SAME
     .xml file that harness_rl/config.py points at  <-- the usual mistake
  2. run it until the harness is fully formed (boxes seated, constraints on)
  3. at that moment call:
         from harness_rl import save_snapshot
         save_snapshot(self.model, self.data, r"<path printed below>")
=============================================================================
"""

import os
import numpy as np
import mujoco as mj

from harness_rl.config import XML, SNAPSHOT
from harness_rl.env import S0_SPEC

print("=" * 70)
print("SNAPSHOT CHECK")
print("=" * 70)

# --- what the model needs ---
if not os.path.isfile(XML):
    raise SystemExit(f"XML not found: {XML}")
model = mj.MjModel.from_xml_path(XML)
need = mj.mj_stateSize(model, S0_SPEC)
print(f"model    : {XML}")
print(f"           nq={model.nq}  nv={model.nv}  neq={model.neq}")
print(f"           -> a snapshot for THIS model must have {need} values "
      f"({need * 8 + 128} bytes on disk)")
print()

# --- what the snapshot has ---
if not os.path.isfile(SNAPSHOT):
    raise SystemExit(f"snapshot not found: {SNAPSHOT}")
have = np.load(SNAPSHOT).size
size = os.path.getsize(SNAPSHOT)
import datetime
when = datetime.datetime.fromtimestamp(os.path.getmtime(SNAPSHOT))
print(f"snapshot : {SNAPSHOT}")
print(f"           {have} values, {size} bytes, last modified {when:%Y-%m-%d %H:%M}")
print()

print("=" * 70)
if have == need:
    print("MATCH — this snapshot belongs to this model. You are good to go.")
else:
    print(f"MISMATCH — the snapshot has {have} values but this model needs {need}.")
    print()
    print("  The snapshot was captured from a DIFFERENT model.")
    print("  Check the XML path inside your prestage script: it must be")
    print(f"      {XML}")
    print("  Then re-run it and call save_snapshot() at the harness-formed moment,")
    print(f"  writing to:  {SNAPSHOT}")
print("=" * 70)
