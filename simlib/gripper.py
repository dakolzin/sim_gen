from __future__ import annotations
import time
import numpy as np
import mujoco

from . import config as cfg
from .mujoco_io import MjContext

# ───── кеши ID / адресов / параметров ────────────────────────────────────────
_GRIP_ACT_ID    = None
_FINGER1_JID    = None
_FINGER2_JID    = None
_FINGER1_QADR   = None
_FINGER2_QADR   = None
_GRIP_LOAD_SID  = None
_GRIP_LOAD_ADR  = None
_CTRL_MIN       = None
_CTRL_MAX       = None
_CTRL_DIR       = None   # +1.0 или -1.0: шаг по u, который УМЕНЬШАЕТ ширину


# ─────────────────────────── служебные инициализаторы ───────────────────────
def _init_ids(ctx: MjContext):
    """Найти ID актуатора и адреса суставов пальцев (один раз)."""
    global _GRIP_ACT_ID, _FINGER1_JID, _FINGER2_JID, _FINGER1_QADR, _FINGER2_QADR

    if _GRIP_ACT_ID is None:
        name = cfg.GRIPPER_ACT_NAME or ""
        if name:
            _GRIP_ACT_ID = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        else:
            _GRIP_ACT_ID = ctx.model.nu - 1  # последний актуатор

    # сустава(ов) пальцев может не быть — это ок
    if _FINGER1_JID is None:
        try:
            if cfg.GRIPPER_FINGER1_JOINT:
                _FINGER1_JID = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, cfg.GRIPPER_FINGER1_JOINT)
            else:
                _FINGER1_JID = -1
        except Exception:
            _FINGER1_JID = -1

    if _FINGER2_JID is None:
        try:
            if cfg.GRIPPER_FINGER2_JOINT:
                _FINGER2_JID = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, cfg.GRIPPER_FINGER2_JOINT)
            else:
                _FINGER2_JID = -1
        except Exception:
            _FINGER2_JID = -1

    # адреса qpos для суставов (если есть)
    _FINGER1_QADR = int(ctx.model.jnt_qposadr[_FINGER1_JID]) if _FINGER1_JID >= 0 else None
    _FINGER2_QADR = int(ctx.model.jnt_qposadr[_FINGER2_JID]) if _FINGER2_JID >= 0 else None


def _init_force_sensor(ctx: MjContext):
    """Ленивая инициализация датчика силы (если задан в конфиге)."""
    global _GRIP_LOAD_SID, _GRIP_LOAD_ADR
    if _GRIP_LOAD_SID is not None:
        return
    try:
        if cfg.GRIPPER_SENSOR_NAME:
            _GRIP_LOAD_SID = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SENSOR, cfg.GRIPPER_SENSOR_NAME)
            _GRIP_LOAD_ADR = int(ctx.model.sensor_adr[_GRIP_LOAD_SID])
        else:
            _GRIP_LOAD_SID = -1
            _GRIP_LOAD_ADR = None
    except Exception:
        _GRIP_LOAD_SID = -1
        _GRIP_LOAD_ADR = None


def _init_ctrl_range(ctx: MjContext):
    """Считать/задать диапазон уставки позиционного актуатора (ctrlrange)."""
    global _CTRL_MIN, _CTRL_MAX
    if _CTRL_MIN is not None and _CTRL_MAX is not None:
        return

    if cfg.GRIPPER_CTRL_MIN is not None and cfg.GRIPPER_CTRL_MAX is not None:
        _CTRL_MIN = float(cfg.GRIPPER_CTRL_MIN)
        _CTRL_MAX = float(cfg.GRIPPER_CTRL_MAX)
        return

    _init_ids(ctx)
    try:
        rng = ctx.model.actuator_ctrlrange[_GRIP_ACT_ID]
        _CTRL_MIN = float(rng[0])
        _CTRL_MAX = float(rng[1])
    except Exception:
        # fallback: исходя из ширины: width = 2*u → уставка половины ширины
        _CTRL_MIN = 0.0
        _CTRL_MAX = 0.5 * float(cfg.GRIPPER_OPEN_M)


def _ensure_ctrl_dir(ctx: MjContext):
    """
    Определить знак _CTRL_DIR так, чтобы u += _CTRL_DIR*du УМЕНЬШАЛ ширину.
    Если в конфиге указан GRIPPER_CTRL_SIGN, используем его без автодетекта.
    """
    global _CTRL_DIR
    if _CTRL_DIR is not None:
        return
    if cfg.GRIPPER_CTRL_SIGN in (+1, 1.0, -1, -1.0):
        _CTRL_DIR = float(np.sign(cfg.GRIPPER_CTRL_SIGN))
        return

    _init_ctrl_range(ctx); _init_ids(ctx)

    u0 = float(ctx.data.ctrl[_GRIP_ACT_ID])
    w0 = gripper_width(ctx)

    eps = 1e-4  # маленький тестовый шаг уставки
    # тест "+eps"
    u_plus = float(np.clip(u0 + eps, _CTRL_MIN, _CTRL_MAX))
    ctx.data.ctrl[_GRIP_ACT_ID] = u_plus
    mujoco.mj_step(ctx.model, ctx.data)
    w_plus = gripper_width(ctx)
    # вернуть
    ctx.data.ctrl[_GRIP_ACT_ID] = u0
    mujoco.mj_step(ctx.model, ctx.data)
    # тест "-eps"
    u_minus = float(np.clip(u0 - eps, _CTRL_MIN, _CTRL_MAX))
    ctx.data.ctrl[_GRIP_ACT_ID] = u_minus
    mujoco.mj_step(ctx.model, ctx.data)
    w_minus = gripper_width(ctx)
    # восстановить
    ctx.data.ctrl[_GRIP_ACT_ID] = u0
    mujoco.mj_step(ctx.model, ctx.data)

    if w_plus < w0 - 1e-6:
        _CTRL_DIR = +1.0
    elif w_minus < w0 - 1e-6:
        _CTRL_DIR = -1.0
    else:
        # не различили — возьмём «классический» минус
        _CTRL_DIR = -1.0


# ─────────────────────────── низкоуровневое управление ──────────────────────
def gripper_ctrl(ctx: MjContext, u: float):
    """Записать сигнал (м) в позиционный актуатор по сухожилию."""
    _init_ids(ctx); _init_ctrl_range(ctx)
    ctx.data.ctrl[_GRIP_ACT_ID] = float(np.clip(u, _CTRL_MIN, _CTRL_MAX))


def _width_to_ctrl(w: float) -> float:
    """Пересчёт ширины (между губками) в уставку (u = 0.5 * width)."""
    return float(np.clip(0.5 * w, 0.0, 0.5 * float(cfg.GRIPPER_OPEN_M)))


def _ctrl_to_width(u: float) -> float:
    """Обратный пересчёт (width = 2*u)."""
    return float(2.0 * u)


# ───────────────────────────── измерение состояния ───────────────────────────
def gripper_width(ctx: MjContext) -> float:
    """Текущая ширина губок (м). По суставам (если есть), иначе по ctrl."""
    _init_ids(ctx)
    if cfg.GRIPPER_WIDTH_SOURCE == "joints" and _FINGER1_QADR is not None:
        q1 = float(ctx.data.qpos[_FINGER1_QADR])
        q2 = float(ctx.data.qpos[_FINGER2_QADR]) if _FINGER2_QADR is not None else q1
        return q1 + q2
    # fallback через ctrl
    return _ctrl_to_width(float(ctx.data.ctrl[_GRIP_ACT_ID]))


def gripper_force(ctx: MjContext) -> float:
    """
    Модуль силы (Н) с датчика (если есть), иначе |actuator_force|.
    """
    _init_ids(ctx); _init_force_sensor(ctx)
    if _GRIP_LOAD_SID is not None and _GRIP_LOAD_SID >= 0 and _GRIP_LOAD_ADR is not None:
        try:
            return abs(float(ctx.data.sensordata[_GRIP_LOAD_ADR]))
        except Exception:
            pass
    try:
        return abs(float(ctx.data.actuator_force[_GRIP_ACT_ID]))
    except Exception:
        return 0.0


# ───────────────────────────── высокоуровневые действия ──────────────────────
def gripper_set(ctx: MjContext,
                width_m: float,
                nsteps: int = cfg.GRIPPER_MOVE_STEPS,
                sleep: float = cfg.GRIPPER_MOVE_SLEEP,
                viewer=None):
    """Перевести ЗУ к требуемой ширине по позиционной уставке с прогоном симуляции."""
    u = _width_to_ctrl(width_m)
    gripper_ctrl(ctx, u)
    nsteps = max(0, int(nsteps))
    for _ in range(nsteps):
        mujoco.mj_step(ctx.model, ctx.data)
        if viewer:
            viewer.sync()
        if sleep > 0.0:
            time.sleep(sleep)


def gripper_open(ctx: MjContext, viewer=None):
    """Открыть до cfg.GRIPPER_OPEN_M."""
    gripper_set(ctx, float(cfg.GRIPPER_OPEN_M), viewer=viewer)


def gripper_close(ctx: MjContext, viewer=None):
    """Закрыть до нуля (без контроля силы)."""
    gripper_set(ctx, float(cfg.GRIPPER_CLOSE_M), viewer=viewer)


def gripper_is_closed(ctx: MjContext,
                      thresh: float = cfg.GRIPPER_CLOSE_THRESH) -> bool:
    """Считать ЗУ закрытым, если ширина ≤ thresh."""
    return gripper_width(ctx) <= float(thresh)


# ───────────────────────── силозависимое закрытие ────────────────────────────
def gripper_close_until(ctx: MjContext,
                        f_thresh: float,
                        *,
                        min_width: float | None = None,
                        timeout_s: float = 3.0,
                        speed_mps: float | None = None,
                        viewer=None) -> float:
    """
    Закрывать ЗУ с постоянной линейной скоростью до достижения порога силы
    ИЛИ пока ширина не станет ≤ min_width, ИЛИ по таймауту.

    Аргументы:
      f_thresh  – порог силы (Н) по датчику 'grip_load' (или по fallback).
      min_width – минимальная ширина, ниже которой не сжимать (м). Можно None.
      timeout_s – ограничение по времени (сек).
      speed_mps – скорость схождения губок (м/с) для ШИРИНЫ; если None → cfg.GRIPPER_SPEED_MPS.

    Возвращает:
      Финальную ширину (м).
    """
    _init_ids(ctx); _init_force_sensor(ctx); _init_ctrl_range(ctx); _ensure_ctrl_dir(ctx)

    # базовые параметры
    dt = float(ctx.model.opt.timestep)
    v = float(speed_mps) if (speed_mps is not None and speed_mps > 0.0) else float(cfg.GRIPPER_SPEED_MPS)
    # ширина = 2 * u  →  du = 0.5 * d(width)
    du = 0.5 * v * dt

    t0 = time.time()
    while (time.time() - t0) < float(timeout_s):
        F = gripper_force(ctx)
        w = gripper_width(ctx)

        if F >= float(f_thresh):
            break
        if min_width is not None and w <= float(min_width) + 1e-4:
            break

        # шаг уставки в ТУ сторону, которая уменьшает ширину
        u = float(ctx.data.ctrl[_GRIP_ACT_ID])
        new_u = float(np.clip(u + _CTRL_DIR * du, _CTRL_MIN, _CTRL_MAX))
        if abs(new_u - u) < 1e-12:
            break

        gripper_ctrl(ctx, new_u)
        mujoco.mj_step(ctx.model, ctx.data)
        if viewer:
            viewer.sync()

    # короткая стабилизация после контакта/таймаута
    for _ in range(4):
        mujoco.mj_step(ctx.model, ctx.data)
        if viewer:
            viewer.sync()

    return gripper_width(ctx)
