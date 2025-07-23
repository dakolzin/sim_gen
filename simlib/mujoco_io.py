"""
Загрузка MuJoCo модели, доступ к трансформам (extrinsics), положениям и ориентациям тел/сайтов,
и выбор служебных ID (камера, база, tcp_site, объекты для сверки).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import numpy as np
import mujoco
from spatialmath import SE3
from . import config as cfg
from .transforms import make_se3, R_GL2CV, ortho_project

@dataclass
class MjContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    cam_id: int
    base_id: int
    tcp_site_id: int
    obj_ids: List[int]
    grasp_geom_ids: Dict[str, int] | None = None

def load_model(xml_path: Path = cfg.XML) -> MjContext:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)

    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.CAM_NAME)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   cfg.BASE_BODY)
    try:
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.TCP_SITE)
    except Exception:
        tcp_site_id = -1

    obj_ids = [i for i in range(model.nbody)
               if 'fractured' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,i) or '')]

    return MjContext(model, data, cam_id, base_id, tcp_site_id, obj_ids)

def load_model_gg(xml_path: Path = cfg.XML) -> MjContext:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)

    grasp_geom_ids = {
        axis: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"grasp_{axis}")
        for axis in ("x", "y", "z")
    }

    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.CAM_NAME)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   cfg.BASE_BODY)
    try:
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.TCP_SITE)
    except Exception:
        tcp_site_id = -1

    obj_ids = [i for i in range(model.nbody)
               if 'fractured' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,i) or '')]

    ctx = MjContext(model, data, cam_id, base_id, tcp_site_id, obj_ids)
    ctx.grasp_geom_ids = grasp_geom_ids          # 👈 так безопаснее
    return ctx

# ── helpers ───────────────────────────────────────────────────────────────────
def mj_forward_if_needed(ctx: MjContext):
    mujoco.mj_forward(ctx.model, ctx.data)

def T_w_b(ctx: MjContext) -> SE3:  # base→world
    mj_forward_if_needed(ctx)
    Rwb = np.array(ctx.data.xmat[ctx.base_id]).reshape(3,3)
    twb = np.array(ctx.data.xpos[ctx.base_id])
    return make_se3(Rwb, twb)

def T_w_c(ctx: MjContext) -> SE3:  # camGL→world
    mj_forward_if_needed(ctx)
    Rwc = np.array(ctx.data.cam_xmat[ctx.cam_id]).reshape(3,3)
    twc = np.array(ctx.data.cam_xpos[ctx.cam_id])
    return make_se3(Rwc, twc)

def T_b_c_gl(ctx: MjContext) -> SE3:
    return T_w_b(ctx).inv() * T_w_c(ctx)

def T_c_b_gl(ctx: MjContext) -> SE3:
    return T_b_c_gl(ctx).inv()

def obj_in_cam_cv(ctx: MjContext, bid: int) -> SE3:
    """Положение и ориентация тела bid в кадре камеры (OpenCV)."""
    R_w_o = np.array(ctx.data.xmat[bid]).reshape(3,3)
    t_w_o = np.array(ctx.data.xpos[bid])
    T_w_o = make_se3(R_w_o, t_w_o)
    T_w_cgl = T_w_c(ctx)
    T_cgl_o = T_w_cgl.inv() * T_w_o   # obj в camGL
    R_c_cv = ortho_project(R_GL2CV @ T_cgl_o.R)
    t_c_cv = R_GL2CV @ T_cgl_o.t
    return make_se3(R_c_cv, t_c_cv)

def tcp_pose_in_base(ctx: MjContext) -> SE3:
    """Фактическое положение и ориентация tcp_site в базовой СК."""
    if ctx.tcp_site_id < 0:
        raise RuntimeError("tcp_site не найден в MuJoCo модели.")
    mj_forward_if_needed(ctx)
    R_w = np.array(ctx.data.site_xmat[ctx.tcp_site_id]).reshape(3,3)
    t_w = np.array(ctx.data.site_xpos[ctx.tcp_site_id])
    return T_w_b(ctx).inv() * make_se3(R_w, t_w)

def body_pose_in_base(ctx: MjContext, bid: int) -> SE3:
    """Положение и ориентация тела bid в базовой СК."""
    mj_forward_if_needed(ctx)
    R_w_o = np.array(ctx.data.xmat[bid]).reshape(3,3)
    t_w_o = np.array(ctx.data.xpos[bid])
    return T_w_b(ctx).inv() * make_se3(R_w_o, t_w_o)
