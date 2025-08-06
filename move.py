#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, sys, numpy as np, mujoco
from mujoco import viewer
from pathlib import Path
from spatialmath import SE3, UnitQuaternion
from roboticstoolbox.robot.ERobot import ERobot

# ---------- файлы ----------------------------------------------------------
SCENE_MJCF = "scene/scene.xml"
URDF_PATH  = str(Path("urdf_out/panda.urdf").resolve())

# ---------- целевой TCP ----------------------------------------------------
POS  = np.array([0.454, 0.169, 0.034])          # м
QUAT = np.array([0.1891, 0.9649, 0.1813, -0.0203])  # x y z w
T_GOAL = SE3.Rt(UnitQuaternion(np.roll(QUAT, 1)).R, POS)

STEPS, DT = 150, 1/60

# ---------- загружаем MuJoCo сцену -----------------------------------------
model = mujoco.MjModel.from_xml_path(SCENE_MJCF)
data0 = mujoco.MjData(model);  mujoco.mj_forward(model, data0)

id_tcp  = model.site("tcp_site").id
id_hand = model.body("hand").id

def ortho(R):
    U, _, Vt = np.linalg.svd(R)
    if np.linalg.det(U @ Vt) < 0: U[:, -1] *= -1
    return U @ Vt

T_hand = SE3.Rt(ortho(data0.xmat[id_hand].reshape(3,3)), data0.xpos[id_hand])
T_tcp  = SE3.Rt(ortho(data0.site_xmat[id_tcp].reshape(3,3)), data0.site_xpos[id_tcp])

# ---------- RTB‑робот -------------------------------------------------------
panda = ERobot.URDF(URDF_PATH)
tcp_link = panda.link_dict["tcp"]
panda._ee_links = [tcp_link] 

# ---------- IK --------------------------------------------------------------
sol = panda.ikine_LM(T_GOAL, q0=np.zeros(7))
if not sol.success: sys.exit(sol.reason)
q_target = sol.q
print("IK OK (rad):", np.round(q_target, 3))

traj = np.linspace(np.zeros(7), q_target, STEPS)

# ---------- индексы суставов без предупреждений ----------------------------
JADR = [model.joint(f"joint{i}").qposadr.item() for i in range(1, 8)]

# ---------- проигрываем траекторию в MuJoCo --------------------------------
data = mujoco.MjData(model)
with viewer.launch_passive(model, data) as v:
    for qk in traj:
        for adr, q in zip(JADR, qk):
            data.qpos[adr] = q
        mujoco.mj_forward(model, data)
        v.sync();  time.sleep(DT)

    # ---------- проверка TCP ------------------------------------------------
    tcp_pos = data.site_xpos[id_tcp]
    R_tcp   = ortho(data.site_xmat[id_tcp].reshape(3, 3))
    tcp_quat = UnitQuaternion(R_tcp).vec     # (x y z w)

    goal_uq  = UnitQuaternion(np.roll(QUAT, 1))
    err_mm   = np.linalg.norm(tcp_pos - POS) * 1e3
    err_ang  = np.degrees(UnitQuaternion(tcp_quat).angdist(goal_uq))

    print(f"\nTCP_base_sim: pos {tcp_pos.round(3)} quat {tcp_quat.round(4)}")
    print(f"Δpos = {err_mm:.2f} mm,  Δangle = {err_ang:.2f} °")

    time.sleep(3)
