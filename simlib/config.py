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
ROOT  = _THIS.parents[1]        
XML   = ROOT / "scene" / "scene.xml"
ASSETS= ROOT / "assets"

# --- Timing ------------------------------------------------
SLEEP_BETWEEN = 1.0
LOOP_SLEEP_SEC = SLEEP_BETWEEN

# ── идентификаторы модели ─────────────────────────────────────────────────────
CAM_NAME   = "rs_d435"
BASE_BODY  = "link0"
TCP_SITE   = "tcp_site"

# ── визуал ────────────────────────────────────────────────────────────────────
WIDTH  = 640
HEIGHT = 480

# ── захватное устройство ──────────────────────────────────────────────────────
GRIPPER_ACT_NAME      = "actuator8"
GRIPPER_OPEN_M        = 0.08
GRIPPER_CLOSE_M       = 0.0
GRIPPER_CLOSE_THRESH  = 0.002
GRIPPER_MOVE_STEPS    = 100
GRIPPER_MOVE_SLEEP    = 0.0
GRIPPER_SPEED_MPS     = 0.04

GRIPPER_FINGER1_JOINT = "finger_joint1"
GRIPPER_FINGER2_JOINT = "finger_joint2"
GRIPPER_SENSOR_NAME   = "grip_load"       # или None
GRIPPER_WIDTH_SOURCE  = "ctrl"          # "joints" | "ctrl"
GRIPPER_CTRL_MIN      = None
GRIPPER_CTRL_MAX      = None
GRIPPER_CTRL_SIGN     = -1.0

# ── сетевой SBG сервер ────────────────────────────────────────────────────────
SBG_HOST = "127.0.0.1"
SBG_PORT = 6000

# ── время ОЖИДАНИЯ ────────────────────────────────────────────────────────────
SETTLE_SEC = 10.0

# ── калибровка ────────────────────────────────────────────────────────────────
APPLY_R_CORR      = False
USE_STATIC_OFFSET = False
R_CORR_MODE = 'left'
R_CORR = np.array([
    [ 0.92901315,  0.35273410,  0.11186247],
    [ 0.26558938, -0.84607179,  0.46219564],
    [ 0.25767584, -0.39967635, -0.87969425],
], dtype=float)
G_OFFSET_CV = np.array([0, 0, 0.029], dtype=float)
DEPTH_SHIFT = 0.0
CAL_SAMPLES_MAX = 25

# ── IK / движение ─────────────────────────────────────────────────────────────
DO_IK           = True
TELEPORT_JOINTS = False
IK_MOVE_DUR     = 2.0
IK_SETTLE_T     = 0.20
USE_TCP_TIP     = False
TCP2TIP_Z       = 0.029

# ── группы для раскрытия осей TCP ─────────────────────────────────────────────
SHOW_TCP_AXES_GEOMGROUP = 4

# ── GRASP / PIPELINE ──────────────────────────────────────────────────────────
PREOPEN_EXTRA_M    = 0.01
CLOSE_MARGIN_M     = 0.002
FALLBACK_PREOPEN   = GRIPPER_OPEN_M
FALLBACK_CLOSE     = GRIPPER_CLOSE_M

F_THRESH           = 150.0
SUCCESS_FORCE_N    = 40.0
MIN_CLOSE_GAIN_M   = 0.006

PRE_OFFSET_M       = 0.15
RET_OFFSET_M       = 0.12

SIDE_OFFSET_BASE   = np.array([-0.20, -0.30, 0.00], dtype=float)
SIDE_Z_DELTAS      = [0.10, -0.10, 0.18, -0.18]
SIDE_XY_SCALES     = [1.0, 1.3, 0.7]
SIDE_PLAN_STEPS    = 40
SIDE_MOVE_TIME     = 1.0

# Сетевые и «пустые» лимиты
NET_FAIL_LIMIT     = 5       # только реальные сетевые ошибки/исключения
IDLE_LIMIT         = 5       # подряд пустых итераций (recv None или 0 grasps)

# ── удобные флаги запуска ─────────────────────────────────────────────────────
@dataclass
class RunFlags:
    do_ik: bool = DO_IK
    move_dur: float = IK_MOVE_DUR
    settle_t: float = IK_SETTLE_T
    show_tcp_axes: bool = True

DEFAULT_FLAGS = RunFlags()
