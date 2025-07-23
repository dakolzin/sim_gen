from __future__ import annotations
import time
import numpy as np
import mujoco

from . import config as cfg
from .mujoco_io import MjContext

# ───── кеши ID / адресов  ────────────────────────────────────────────────────
_GRIP_ACT_ID   = None
_FINGER1_JID   = None
_FINGER2_JID   = None
_FINGER1_QADR  = None
_FINGER2_QADR  = None
_GRIP_LOAD_SID = None
_GRIP_LOAD_ADR = None


def _init_ids(ctx: MjContext):
    """Найти ID/адреса один раз."""
    global _GRIP_ACT_ID, _FINGER1_JID, _FINGER2_JID, _FINGER1_QADR, _FINGER2_QADR

    if _GRIP_ACT_ID is None:
        _GRIP_ACT_ID = mujoco.mj_name2id(
            ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, cfg.GRIPPER_ACT_NAME
        ) if cfg.GRIPPER_ACT_NAME else ctx.model.nu - 1

    if _FINGER1_JID is None:
        try:
            _FINGER1_JID = mujoco.mj_name2id(
                ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"
            )
        except Exception:
            _FINGER1_JID = -1
    if _FINGER2_JID is None:
        try:
            _FINGER2_JID = mujoco.mj_name2id(
                ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"
            )
        except Exception:
            _FINGER2_JID = -1

    if _FINGER1_JID >= 0:
        _FINGER1_QADR = ctx.model.jnt_qposadr[_FINGER1_JID]
    if _FINGER2_JID >= 0:
        _FINGER2_QADR = ctx.model.jnt_qposadr[_FINGER2_JID]


def _init_force_sensor(ctx: MjContext):
    """Ленивая инициализация датчика grip_load."""
    global _GRIP_LOAD_SID, _GRIP_LOAD_ADR
    if _GRIP_LOAD_SID is None:
        try:
            _GRIP_LOAD_SID = mujoco.mj_name2id(
                ctx.model, mujoco.mjtObj.mjOBJ_SENSOR, "grip_load"
            )
            _GRIP_LOAD_ADR = ctx.model.sensor_adr[_GRIP_LOAD_SID]
        except Exception:
            _GRIP_LOAD_SID = -1


# ───── низкоуровневое управление ─────────────────────────────────────────────
def gripper_ctrl(ctx: MjContext, u: float):
    """Записать сигнал (м) в позиционный актуатор по сухожилию."""
    _init_ids(ctx)
    ctx.data.ctrl[_GRIP_ACT_ID] = float(u)

def _width_to_ctrl(w: float) -> float:
    # ширина = q1 + q2, а позиционный актуатор задаёт ход КАЖДОГО пальца
    return float(np.clip(w * 0.5, 0.0, cfg.GRIPPER_OPEN_M * 0.5))

def _ctrl_to_width(u: float) -> float:
    return float(u * 2.0)


# ───── измерение состояния ───────────────────────────────────────────────────
def gripper_width(ctx: MjContext) -> float:
    """Текущая ширина пальцев (м)."""
    _init_ids(ctx)
    if _FINGER1_QADR is not None:
        q1 = ctx.data.qpos[_FINGER1_QADR]
        q2 = ctx.data.qpos[_FINGER2_QADR] if _FINGER2_QADR is not None else q1
        return float(q1 + q2)
    # fallback через ctrl
    return _ctrl_to_width(ctx.data.ctrl[_GRIP_ACT_ID])

def gripper_force(ctx: MjContext) -> float:
    """Модуль силы (Н) с датчика grip_load, 0.0 если сенсор отсутствует."""
    _init_force_sensor(ctx)
    return abs(ctx.data.sensordata[_GRIP_LOAD_ADR]) if _GRIP_LOAD_SID >= 0 else 0.0

# ───── высокоуровневые действия ──────────────────────────────────────────────
def gripper_set(ctx, width_m, nsteps=cfg.GRIPPER_MOVE_STEPS, sleep=cfg.GRIPPER_MOVE_SLEEP, viewer=None):
    u = _width_to_ctrl(width_m)
    gripper_ctrl(ctx, u)
    for i in range(max(0, nsteps)):
        mujoco.mj_step(ctx.model, ctx.data)
        if viewer: viewer.sync()
        if i % 5 == 0:   # реже, чтобы не засорять
            q1 = ctx.data.qpos[_FINGER1_QADR]
            q2 = ctx.data.qpos[_FINGER2_QADR]
            print(f'[DBG] step {i}: ctrl={u:.4f}  q1={q1:.4f}  q2={q2:.4f}  width={q1+q2:.4f}')
        if sleep > 0: time.sleep(sleep)

def gripper_open(ctx, viewer=None):
    gripper_set(ctx, cfg.GRIPPER_OPEN_M, viewer=viewer)

def gripper_close(ctx, viewer=None):
    gripper_set(ctx, cfg.GRIPPER_CLOSE_M, viewer=viewer)  # CLOSE_M = 0.0

def gripper_is_closed(ctx: MjContext,
                      thresh: float = cfg.GRIPPER_CLOSE_THRESH) -> bool:
    return gripper_width(ctx) <= thresh


# ───── силомоментное закрытие ───────────────────────────────────────────
def gripper_close_until(ctx,
                        f_thresh: float,
                        step_ctrl: float = 5e-4,
                        max_iters: int = 2000,
                        viewer=None) -> float:
    _init_ids(ctx); _init_force_sensor(ctx)

    for it in range(max_iters):
        F   = gripper_force(ctx)
        w   = gripper_width(ctx)
        u   = ctx.data.ctrl[_GRIP_ACT_ID]
        print(f"[DBG] force-loop: it={it} w={w:.4f} F={F:.1f} ctrl={u:.5f}")

        if F >= f_thresh:
            break

        # чуть уменьшить зазор
        new_u = max(u - step_ctrl, 0.0)
        if new_u >= u - 1e-9:
            print("[WARN] ctrl can't be reduced -> break")
            break

        gripper_ctrl(ctx, new_u)
        for _ in range(4):
            mujoco.mj_step(ctx.model, ctx.data)
        if viewer: viewer.sync()

    return gripper_width(ctx)