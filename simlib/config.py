"""
Глобальный конфиг симуляции / калибровки.

Измени здесь пути, имена камеры, матрицу коррекции, офсеты и режимы.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ── базовые пути ──────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
ROOT  = _THIS.parents[1]          # sim_d435/
XML   = ROOT / "scene" / "scene.xml"
ASSETS= ROOT / "assets"           # на всякий случай (не всегда нужно)

# --- Timing ------------------------------------------------
SLEEP_BETWEEN = 1.0               # сек — пауза главного цикла (раньше использовалось в sim_transform)
LOOP_SLEEP_SEC = SLEEP_BETWEEN    # алиас для читаемости

# ── идентификаторы модели ─────────────────────────────────────────────────────
CAM_NAME   = "rs_d435"
BASE_BODY  = "link0"
TCP_SITE   = "tcp_site"           

# ── визуал ────────────────────────────────────────────────────────────────────
WIDTH  = 640
HEIGHT = 480

# ── захватное устройство ─────────────────────────────────────────────────────────────────
GRIPPER_ACT_NAME      = "actuator8"   # имя позиционного актуатора (тендон)
GRIPPER_OPEN_M        = 0.08          # м, полное раскрытие (2 * 0.04)
GRIPPER_CLOSE_M       = 0.0           # м, полностью закрыт
GRIPPER_CLOSE_THRESH  = 0.002         # м, считать закрытым если ширина < ...
GRIPPER_MOVE_STEPS    = 100           # шагов симуляции при движении (если viewer не None)
GRIPPER_MOVE_SLEEP    = 0.0           # пауза между шагами (0 = быстро)
GRIPPER_SPEED_MPS     = 0.04          # скорость схождения (м/с) для ШИРИНЫ

# Имена суставов пальцев (если есть прямой доступ к их qpos):
GRIPPER_FINGER1_JOINT = "finger_joint1"
GRIPPER_FINGER2_JOINT = "finger_joint2"   # можно оставить None, если один палец симметрируется

# Имя датчика силы (если есть). Если None — используем actuator_force:
GRIPPER_SENSOR_NAME   = "grip_load"       # или None, если сенсора нет

# Источник текущей ширины: 'joints' (предпочтительно) или 'ctrl'
GRIPPER_WIDTH_SOURCE  = "joints"

# Диапазон уставки позиционного актуатора (если хотите задать явно).
# Если оба None — читается из actuator_ctrlrange.
GRIPPER_CTRL_MIN      = None   # например 0.0
GRIPPER_CTRL_MAX      = None   # например 0.04 (это ход ОДНОГО пальца)

# Направление шага уставки, которое УМЕНЬШАЕТ ширину:
#   None  – определить автоматически (рекомендуется);
#   +1.0  – u+=du уменьшает ширину;
#   -1.0  – u-=du уменьшает ширину.
GRIPPER_CTRL_SIGN     = None

# ── сетевой SBG сервер ────────────────────────────────────────────────────────
SBG_HOST = "127.0.0.1"
SBG_PORT = 6000

# ── время ОЖИДАНИЯ ────────────────────────────────────────────────────────────
SETTLE_SEC = 10.0

# ── калибровка (из твоего RUN2) ───────────────────────────────────────────────
APPLY_R_CORR      = False
USE_STATIC_OFFSET = False

R_CORR_MODE = 'left'

R_CORR = np.array([
    [ 0.92901315,  0.35273410,  0.11186247],
    [ 0.26558938, -0.84607179,  0.46219564],
    [ 0.25767584, -0.39967635, -0.87969425],
], dtype=float)

G_OFFSET_CV = np.array([0, 0, 0.029], dtype=float)  # м

DEPTH_SHIFT = 0.0  # множитель depth вдоль +Z гриппера (после коррекции)

CAL_SAMPLES_MAX = 25

# ── IK / движение ─────────────────────────────────────────────────────────────
DO_IK           = True          # по умолчанию включен (скрипты могут переопределять)
TELEPORT_JOINTS = False         # если True — мгновенно ставим qpos
IK_MOVE_DUR     = 2.0           # сек (траектория линейная в конфиг-пространстве)
IK_SETTLE_T     = 0.20          # сек ожидания после прихода в цель
USE_TCP_TIP     = False         # использовать ли TCP→TIP смещение
TCP2TIP_Z       = 0.029           # м 

# ── группы для раскрытия осей TCP ─────────────────────────────────────────────
SHOW_TCP_AXES_GEOMGROUP = 4

# ── полезный dataclass для быстрых overrides ──────────────────────────────────
@dataclass
class RunFlags:
    do_ik: bool = DO_IK
    move_dur: float = IK_MOVE_DUR
    settle_t: float = IK_SETTLE_T
    show_tcp_axes: bool = True

DEFAULT_FLAGS = RunFlags()

