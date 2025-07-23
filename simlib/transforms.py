# simlib/transforms.py
"""
Утилиты преобразований между системами координат.

Кадры:
    CV  : OpenCV камеры (X→, Y↓, Z→вперёд).
    GL  : OpenGL камеры (X→, Y↑, Z← в камеру).
    BASE: База робота.

Главная функция: ``camcv2base()`` — перевод grasp'а сети (t,R в CV) в базовую
СК робота, с учётом:
    • вращательной коррекции cfg.R_CORR (режим cfg.R_CORR_MODE)
    • статического позиционного офсета cfg.G_OFFSET_CV (dv вдоль осей grasp'а)
    • depth shift (cfg.DEPTH_SHIFT)
    • TCP→TIP смещения (аргумент tcp2tip)

ВАЖНО про офсет dv:
  dv измерен в *сырой рамке grasp'а сети* (до ориентационной коррекции).
  Если мы применяем R_CORR, локальные оси меняются. Чтобы dv оставался
  семантически тем же смещением вдоль осей гриппера, нужно выразить dv
  в пост‑корректной рамке. Для режима 'left' (R_corr @ R_cv) это:
      dv_corr = R_corr.T @ dv_raw
  Для 'right' аналогично (т.к. матрицы ортогональны, инверсия = transpose).
"""

from __future__ import annotations
import numpy as np
from spatialmath import SE3, UnitQuaternion, SO3
from . import config as cfg

# --------------------------------------------------------------------------- #
#  CV ↔ GL переходы
# --------------------------------------------------------------------------- #
# OpenCV: X→, Y↓, Z→вперёд;  OpenGL: X→, Y↑, Z← (в камеру)
R_CV2GL = np.diag([1, -1, -1])
R_GL2CV = R_CV2GL  # сам себе инверсия
SWAP_COLS = (0, 2, 1)

# --------------------------------------------------------------------------- #
#  Линейные утилиты
# --------------------------------------------------------------------------- #
def ortho_project(R: np.ndarray) -> np.ndarray:
    """Проецирует произвольную 3×3 на ближайшую ортонормированную SO(3)."""
    R = np.asarray(R, float).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def make_se3(R, t) -> SE3:
    """Создать SE3 из (R,t) с ортогонализацией."""
    Rn = ortho_project(np.asarray(R, float).reshape(3, 3))
    t = np.asarray(t, float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = Rn
    T[:3, 3] = t
    return SE3(T, check=False)


def safe_uq_from_R(R) -> UnitQuaternion:
    """UnitQuaternion из R c ортогонализацией."""
    return UnitQuaternion(SO3(ortho_project(R), check=False))


def safe_uq_from_q(q_xyzw) -> UnitQuaternion:
    """UnitQuaternion из (x,y,z,w) с ортогонализацией."""
    q_xyzw = np.asarray(q_xyzw, float).ravel()
    uq = UnitQuaternion(q_xyzw)
    return UnitQuaternion(SO3(ortho_project(uq.R), check=False))


def rel_angle_deg(Ra, Rb) -> float:
    """Угол поворота Ra→Rb в градусах."""
    Ra = ortho_project(Ra)
    Rb = ortho_project(Rb)
    Rrel = Ra @ Rb.T
    c = (np.trace(Rrel) - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def rel_euler_xyz_deg(Ra, Rb):
    """XYZ Эйлеры Ra→Rb в градусах (для отладочной печати)."""
    Ra = ortho_project(Ra)
    Rb = ortho_project(Rb)
    Rrel = Ra @ Rb.T
    sy = np.sqrt(Rrel[0, 0] ** 2 + Rrel[1, 0] ** 2)
    if sy < 1e-8:  # сингулярность
        x = np.degrees(np.arctan2(-Rrel[1, 2], Rrel[1, 1]))
        y = np.degrees(np.arctan2(-Rrel[2, 0], sy))
        z = 0.0
    else:
        x = np.degrees(np.arctan2(Rrel[2, 1], Rrel[2, 2]))
        y = np.degrees(np.arctan2(-Rrel[2, 0], sy))
        z = np.degrees(np.arctan2(Rrel[1, 0], Rrel[0, 0]))
    return x, y, z


def log_pose(tag: str, T: SE3):
    """Печать позы SE3 в формате pos[...] quat[xyzw]."""
    p = T.t
    q = safe_uq_from_R(T.R).vec
    print(
        f"{tag}: pos [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}] "
        f"quat [{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}]"
    )


# --------------------------------------------------------------------------- #
#  Grasp parsing helpers (дублируются в sbg_client, но оставим для удобства)
# --------------------------------------------------------------------------- #
def parse_grasp_row(row: np.ndarray):
    """
    Универсальный парсер строк grasp'а.
      ≥15: [score, w, h, d, R(9), t(3), (obj_id?)]
       7 : [t(3), quat_xyzw(4)]  -- fallback
    Возврат: (t, R, w, h, d, score, obj_id)
    """
    row = np.asarray(row).ravel()
    n = row.size
    if n >= 15:
        score = float(row[0])
        width = float(row[1])
        height = float(row[2])
        depth = float(row[3])
        R = row[4:13].reshape(3, 3)
        remain = row[13:]
        if remain.size >= 4:
            t = remain[:3]
            obj_id = int(remain[3])
        else:
            t = remain[:3]
            obj_id = -1
        return t, R, width, height, depth, score, obj_id
    elif n == 7:
        t = row[:3]
        q = row[3:7]
        R = safe_uq_from_q(q).R
        return t, R, np.nan, np.nan, np.nan, np.nan, -1
    else:
        raise ValueError(f"grasp row len={n} не поддерживается")


# --------------------------------------------------------------------------- #
#  Коррекции ориентации + офсета
# --------------------------------------------------------------------------- #
def apply_r_corr_cv(R_cv: np.ndarray) -> np.ndarray:
    """
    Применить cfg.R_CORR к ориентации grasp'а в КАДРЕ CV камеры,
    согласно cfg.R_CORR_MODE.
    """
    mode = getattr(cfg, "R_CORR_MODE", "left")
    
    if not cfg.APPLY_R_CORR or mode == "off":
        return ortho_project(R_cv)

    if mode == "left":
        # R_corr ∘ R_cv
        R = cfg.R_CORR @ R_cv
    elif mode == "right":
        # R_cv ∘ R_corr
        R = R_cv @ cfg.R_CORR
    else:
        R = R_cv
    return ortho_project(R)


def transform_offset_for_mode(dv_cv: np.ndarray) -> np.ndarray:
    """
    Переориентировать dv, измеренный в сырой рамке grasp'а сети (до коррекции),
    в рамку ПОСЛЕ применения cfg.R_CORR_MODE.
    """
    dv_cv = np.asarray(dv_cv, float).reshape(3)
    if not cfg.USE_STATIC_OFFSET or not cfg.APPLY_R_CORR:
        return dv_cv

    mode = getattr(cfg, "R_CORR_MODE", "left")

    # Для ортогональных матриц R^-1 = R.T; 'left'/'right' дают одну и ту же
    # формулу для переноса вектора между рамками (речь о локальной смене базиса).
    if mode in ("left", "right"):
        return cfg.R_CORR.T @ dv_cv
    else:
        return dv_cv


def dbg_axes(tag, R):
    gx, gy, gz = R[:, 0], R[:, 1], R[:, 2]
    print(
        f"[DBG {tag}] "
        f"gx={np.array2string(gx, precision=3)} "
        f"gy={np.array2string(gy, precision=3)} "
        f"gz={np.array2string(gz, precision=3)}"
    )


# --------------------------------------------------------------------------- #
#  CV → BASE конверсия grasp'а
# --------------------------------------------------------------------------- #
def camcv2base(
    t_cv,
    R_cv,
    T_b_c_gl: SE3,
    depth=None,
    tcp2tip: SE3 | None = None,
    debug: bool = True,
):
    """
    Конвертировать grasp сети (OpenCV‑камеры) в базу робота.
    Возврат: (G_net, G_tcp)
    """

    # 0) коррекция по R_CORR (если включена)
    R_cv_corr = apply_r_corr_cv(R_cv)                   # (3×3)
    gx_net_cv = R_cv_corr[:, 0]              # вектор подхода сети в CV
    #  В CV‑рамке: +Z = вперёд от камеры,  −Z = в сторону камеры (вниз к столу)
    if gx_net_cv[2] > 0:                     # >0 ⇒ смотрит "вверх"
        # Повернуть рамку на 180° вокруг её оси Z (depth) :
        #   gx → -gx , gy → -gy , gz →  gz
        FLIP_XY = np.diag([-1, -1, 1])
        R_cv_corr = R_cv_corr @ FLIP_XY

    # 1) CV‑рамка сети ➜ рамка TCP Panda  ──►  R_tcp_cv
    #    gx_net (approach)  →  gz_tcp  (approach, вниз)
    #    gy_net (binormal)  →  gy_tcp
    #    gz_net (depth)     → −gx_tcp
    CV2TCP = np.array([[ 0,  0, -1],    # tcp_x = −gz_net
                       [ 0,  1,  0],    # tcp_y =  gy_net
                       [1,  0,  0]],   # tcp_z = −gx_net  (down)
                      dtype=float)
    R_tcp_cv = R_cv_corr @ CV2TCP        #  !!!   post‑multiply  !!!

    # 2) OpenCV ➜ OpenGL
    R_gl = ortho_project(R_CV2GL @ R_tcp_cv)
    t_gl = R_CV2GL @ np.asarray(t_cv, float).reshape(3)

    # 3) статический офсет dv
    if cfg.USE_STATIC_OFFSET:
        t_gl += R_gl @ transform_offset_for_mode(cfg.G_OFFSET_CV)

    # 4) depth‑shift
    if depth is not None and cfg.DEPTH_SHIFT:
        t_gl += R_gl[:, 2] * (cfg.DEPTH_SHIFT * depth)

    # 5) сборка поз
    G_cam  = make_se3(R_gl, t_gl)          # camGL → grasp‑центр
    G_net  = T_b_c_gl * G_cam              # base  → grasp‑центр
    G_tcp  = G_net if tcp2tip is None else (G_net * tcp2tip)

    if debug:
        gz = G_tcp.R[:, 2]
        print("dot(gz_tcp, −Z_world) =", np.dot(gz, [0, 0, -1]).round(3))

    return G_net, G_tcp
