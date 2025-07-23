"""
Решение IK через Robotics Toolbox Panda и движение суставов MuJoCo.
"""
from __future__ import annotations
import time
import numpy as np
import mujoco
from spatialmath import SE3
from . import config as cfg
from .transforms import safe_uq_from_q
from .mujoco_io import MjContext
from . import mujoco_io as mj
try:
    import roboticstoolbox as rtb
    _PANDA = rtb.models.Panda()
except Exception:  # rtb optional
    _PANDA = None

_RTB_TOOL_INIT_DONE = False

def have_rtb():
    return _PANDA is not None

def solve_ik(pos, quat_xyzw):
    if _PANDA is None:
        return None, False
    uq = safe_uq_from_q(quat_xyzw)
    qd, ok, *_ = _PANDA.ik_LM(SE3.Rt(uq.R, pos))
    return qd, bool(ok)

def drive_mujoco(ctx: MjContext, qd, viewer=None, dur=cfg.IK_MOVE_DUR, teleport=cfg.TELEPORT_JOINTS):
    """
    Задать суставы MuJoCo:
      teleport=True  – мгновенная установка qpos; без динамики.
      teleport=False – линейная интерполяция через ctrl (псевдо PD).
    """
    if teleport:
        ctx.data.qpos[:7] = qd
        ctx.data.qvel[:7] = 0
        mujoco.mj_forward(ctx.model, ctx.data)
        if viewer: viewer.sync()
        return

    q0 = ctx.data.ctrl[:7].copy()
    steps = max(2, int(dur / ctx.model.opt.timestep))
    for q in np.linspace(q0, qd, steps):
        ctx.data.ctrl[:7] = q
        mujoco.mj_step(ctx.model, ctx.data)
        if viewer: viewer.sync()
        time.sleep(ctx.model.opt.timestep)

def goto_arm(ctx: MjContext, pos, quat_xyzw, viewer=None, flags=None):
    """
    Обёртка: IK + движение.
    flags: cfg.RunFlags (если None, берём cfg.DEFAULT_FLAGS)
    Возврат: (ok:bool, qd:np.ndarray|None)
    """
    if flags is None: flags = cfg.DEFAULT_FLAGS
    if not flags.do_ik:
        print('[IK disabled]'); return False, None
    if _PANDA is None:
        print('[IK] Robotics Toolbox недоступен'); return False, None

    qd, ok = solve_ik(pos, quat_xyzw)
    if not ok:
        print('[IK] fail'); return False, None

    drive_mujoco(ctx, qd, viewer, dur=flags.move_dur, teleport=flags.teleport)
    return True, qd


def calibrate_rtb_tool_from_mj(ctx):
    """
    Оценивает преобразование от фланца RTB Panda к tcp_site MuJoCo и
    присваивает его _PANDA.tool. Выполнять 1 раз после загрузки модели.
    """
    global _RTB_TOOL_INIT_DONE, _PANDA
    if _PANDA is None or _RTB_TOOL_INIT_DONE:
        return

    # текущие суставы в симуляции
    q = ctx.data.qpos[:7].copy()

    # 1) фланец RTB (тот, что IK оптимизирует) в базе
    T_rtb_flange = _PANDA.fkine(q)   # base->flange

    # 2) tcp_site MuJoCo в базе
    T_mj_tcp = mj.tcp_pose_in_base(ctx)  # base->tcp_site

    # 3) нужен flange->tcp:  T_flange.inv() * T_tcp
    T_tool = T_rtb_flange.inv() * T_mj_tcp

    # назначаем как tool
    _PANDA.tool = T_tool
    _RTB_TOOL_INIT_DONE = True
    print('[IK] calibrated RTB tool from MuJoCo tcp_site:')
    print('      transl =', T_tool.t, '(m)')