"""
simlib/ik_driver.py
-------------------
Обёртка над Robotics Toolbox для решения IK и движения суставов в MuJoCo.

Публичный API:
    have_rtb()                        – доступна ли RTB
    solve_ik(pos, quat_xyzw, q0)      – IK в (м, кватерн. XYZW) с поддержкой семени q0
    goto_arm(ctx, pos, quat_xyzw, …)  – IK + движение (семя = текущие 7 суставов)
    goto_arm_joints(ctx, q_target, …) – прямое движение по суставам
    calibrate_rtb_tool_from_mj(ctx)   – забирает flange→tcp_site из MuJoCo в RTB.tool
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import mujoco
from spatialmath import SE3, UnitQuaternion
from . import config as cfg
from .transforms import safe_uq_from_q
from .mujoco_io import MjContext
from . import mujoco_io as mj

# --------------------------------------------------------------------------- #
# 1) Грузим URDF Panda (звено "tcp" = tcp_site в MJCF)
# --------------------------------------------------------------------------- #
try:
    from roboticstoolbox.robot.ERobot import ERobot
    _URDF_PATH = Path(__file__).resolve().parents[1] / "urdf_out" / "panda.urdf"
    _PANDA = ERobot.URDF(str(_URDF_PATH))
    _PANDA._ee_links = [_PANDA.link_dict["tcp"]]        # конец-эффектор = tcp
except Exception as _e:                                 # RTB не установлена
    print(f'[IK] Robotics Toolbox unavailable: {_e}')
    _PANDA = None


# --------------------------------------------------------------------------- #
# 2) Сервисные функции
# --------------------------------------------------------------------------- #
def have_rtb() -> bool:
    """True если Robotics Toolbox успешно импортирована и Panda загружена."""
    return _PANDA is not None


def solve_ik(pos_xyz, quat_xyzw, q0=None):
    """
    Решить IK: позиция в метрах, кватернион (x y z w, OpenCV-формат).

    Аргументы:
        pos_xyz   – (3,)
        quat_xyzw – (4,) в порядке XYZW
        q0        – семя IK (np.ndarray (7,)). Если None, берутся нули.

    Возврат:
        (qd, ok):
            qd – np.ndarray (7,) или None
            ok – bool, успех
    """
    if _PANDA is None:
        return None, False
    uq = safe_uq_from_q(quat_xyzw)                     # → UnitQuaternion
    q0_seed = np.zeros(7) if q0 is None else np.asarray(q0, float).reshape(7)
    sol = _PANDA.ikine_LM(SE3.Rt(uq.R, pos_xyz), q0=q0_seed)
    return (sol.q if sol.success else None), bool(sol.success)


def _drive_mujoco(ctx: MjContext, qd, viewer=None, *, dur=cfg.IK_MOVE_DUR):
    """
    Линейно интерполировать суставы MuJoCo к вектору qd за время dur (сек).
    • qd может содержать 7 или полный len(qpos) элементов.
    """
    qd = np.asarray(qd, float).ravel()
    n = min(qd.size, ctx.data.ctrl.size)

    q0 = ctx.data.ctrl[:n].copy()
    steps = max(2, int(dur / ctx.model.opt.timestep))
    for q in np.linspace(q0, qd[:n], steps):
        ctx.data.ctrl[:n] = q
        mujoco.mj_step(ctx.model, ctx.data)
        if viewer:
            viewer.sync()
        time.sleep(ctx.model.opt.timestep)


# --------------------------------------------------------------------------- #
# 3) Публичные вызовы
# --------------------------------------------------------------------------- #
def goto_arm_joints(ctx: MjContext, q_target, viewer=None, flags=None):
    """
    Двигаться по суставам (без IK).
    q_target длиной 7 или len(qpos).
    """
    if flags is None:
        flags = cfg.DEFAULT_FLAGS

    q_target = np.asarray(q_target, float).reshape(-1)
    n_all = len(ctx.data.qpos)

    if q_target.size == n_all:          # полный вектор
        q_full = q_target
    elif q_target.size == 7:            # только рука
        q_full = ctx.data.qpos.copy()
        q_full[:7] = q_target
    else:
        raise ValueError(f'Unsupported joint vector len={q_target.size}')

    _drive_mujoco(ctx, q_full, viewer, dur=flags.move_dur)
    return True, q_target


def goto_arm(ctx: MjContext, pos, quat_xyzw, viewer=None, flags=None):
    """
    IK + движение по точке TCP (pos м, quat XYZW).
    Возврат (ok, qd).
    """
    if flags is None:
        flags = cfg.DEFAULT_FLAGS
    if not flags.do_ik:
        print('[IK disabled]'); return False, None
    if _PANDA is None:
        print('[IK] Robotics Toolbox недоступен'); return False, None

    # ВАЖНО: семя = текущие 7 суставов (чтобы не перепрыгивать на другую ветвь IK)
    try:
        q0_seed = ctx.data.qpos[:7].copy()
    except Exception:
        q0_seed = None

    qd, ok = solve_ik(pos, quat_xyzw, q0=q0_seed)
    if not ok:
        print('[IK] solve failed'); return False, None

    _drive_mujoco(ctx, qd, viewer, dur=flags.move_dur)
    time.sleep(flags.settle_t)
    return True, qd


# --------------------------------------------------------------------------- #
# 4) Калибровка flange→tcp из MuJoCo в RTB.tool
# --------------------------------------------------------------------------- #
def calibrate_rtb_tool_from_mj(ctx):
    """
    Берём смещение flange→tcp_site из MuJoCo и пишем его в _PANDA.tool
    """
    if _PANDA is None:
        return

    # --- позиции в MuJoCo ---
    id_l7  = ctx.model.body("link7").id
    id_tcp = ctx.model.site("tcp_site").id

    R_l7  = ctx.data.xmat[id_l7].reshape(3, 3)
    t_l7  = ctx.data.xpos[id_l7]
    R_tcp = ctx.data.site_xmat[id_tcp].reshape(3, 3)
    t_tcp = ctx.data.site_xpos[id_tcp]

    from .transforms import ortho_project, make_se3
    T_l7  = make_se3(ortho_project(R_l7),  t_l7)
    T_tcp = make_se3(ortho_project(R_tcp), t_tcp)

    _PANDA.tool = T_l7.inv() * T_tcp
    print('[IK] calibrated tool:', _PANDA.tool.t)
