"""Логгирование положений и ориентаций, форматирование чисел."""
import numpy as np
from spatialmath import SE3, UnitQuaternion, SO3

def _ortho(R):
    U,_,Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:,-1]*=-1; Rn = U @ Vt
    return Rn

def safe_uq_from_R(R):
    return UnitQuaternion(SO3(_ortho(R), check=False))

def log_pose(tag: str, T: SE3):
    p = T.t
    q = safe_uq_from_R(T.R).vec
    print(f"{tag}: pos [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}] quat [{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}]")

def rel_angle_deg(Ra,Rb):
    Rrel = Ra @ Rb.T
    c = (np.trace(Rrel)-1.0)*0.5
    c = np.clip(c,-1.0,1.0)
    return float(np.degrees(np.arccos(c)))

def rel_euler_xyz_deg(Ra,Rb):
    Rrel = Ra @ Rb.T
    sy = np.sqrt(Rrel[0,0]**2 + Rrel[1,0]**2)
    if sy < 1e-8:
        x = np.degrees(np.arctan2(-Rrel[1,2],Rrel[1,1]))
        y = np.degrees(np.arctan2(-Rrel[2,0],sy))
        z = 0.0
    else:
        x = np.degrees(np.arctan2(Rrel[2,1],Rrel[2,2]))
        y = np.degrees(np.arctan2(-Rrel[2,0],sy))
        z = np.degrees(np.arctan2(Rrel[1,0],Rrel[0,0]))
    return x,y,z
