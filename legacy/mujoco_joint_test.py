#!/usr/bin/env python3
import mujoco
import numpy as np
from pathlib import Path

XML = Path(__file__).resolve().parent / "scene" / "scene.xml"

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print("MJCF joint axes (world) @ q=0:")
for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
    ax_local = m.jnt_axis[j]
    ax_world = d.xaxis[j]
    print(f"{name:>12s}: local={ax_local}, world={ax_world}")
