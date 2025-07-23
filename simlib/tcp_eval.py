"""
Вычисление ошибок между целевой позой TCP (из grasp) и фактической позой в MuJoCo.
"""
from __future__ import annotations
import numpy as np
from spatialmath import SE3
from .logutil import rel_angle_deg

def tcp_error(goal: SE3, actual: SE3):
    """
    Возврат: dict(px,py,pz,dp_mm,da_deg, comp_gx_mm, comp_gy_mm, comp_gz_mm)
    где comp_* — проекции ошибки на оси целевого grasp'а.
    """
    dv = actual.t - goal.t
    dp_mm = float(np.linalg.norm(dv) * 1000.0)
    da_deg = rel_angle_deg(actual.R, goal.R)
    gx,gy,gz = goal.R[:,0], goal.R[:,1], goal.R[:,2]
    return dict(
        dp_mm=dp_mm,
        da_deg=da_deg,
        comp_gx_mm=float(np.dot(dv,gx)*1000.0),
        comp_gy_mm=float(np.dot(dv,gy)*1000.0),
        comp_gz_mm=float(np.dot(dv,gz)*1000.0),
    )
