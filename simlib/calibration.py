"""
Накопление калибровочных выборок и оценка R_CORR + dv (object-grasp).
"""
from __future__ import annotations
import numpy as np

class CalibAccumulator:
    def __init__(self):
        self.Rg = []  # grasp (CV)
        self.Ro = []  # object (CV)
        self.dv = []  # dv в осях grasp (gx,gy,gz)

    def add(self, Rg, Ro, dv_xyz, Rg_axes=None):
        """
        Rg,Ro: (3,3) матрицы ориентаций grasp/obj в CV.
        dv_xyz: obj.t - grasp.t (в CV, XYZ).
        Если Rg_axes=None: вычислим геттер, иначе используем.
        """
        Rg = np.asarray(Rg,float).reshape(3,3)
        Ro = np.asarray(Ro,float).reshape(3,3)
        dv_xyz = np.asarray(dv_xyz,float).reshape(3)

        gx,gy,gz = Rg[:,0], Rg[:,1], Rg[:,2]
        dvx = np.dot(dv_xyz,gx)
        dvy = np.dot(dv_xyz,gy)
        dvz = np.dot(dv_xyz,gz)
        self.Rg.append(Rg.copy())
        self.Ro.append(Ro.copy())
        self.dv.append([dvx,dvy,dvz])

    def done(self, verbose=True):
        """Возврат: (R_corr_est, mean_dv, med_dv)"""
        A = np.zeros((3,3))
        for Rg,Ro in zip(self.Rg,self.Ro):
            A += Ro @ Rg.T
        U,_,Vt = np.linalg.svd(A)
        Rcorr = U @ Vt
        if np.linalg.det(Rcorr) < 0:
            U[:,-1]*=-1; Rcorr = U @ Vt
        dv_arr = np.array(self.dv)
        mean_dv = dv_arr.mean(axis=0)
        med_dv  = np.median(dv_arr,axis=0)
        if verbose:
            print('\n[CAL] ===== DONE (%d samples) =====' % len(self.Rg))
            print('[CAL] R_corr_est =\n', Rcorr)
            print('[CAL] det=%.6f' % np.linalg.det(Rcorr))
            print('[CAL] mean_dv_components (m) =', mean_dv)
            print('[CAL] med_dv_components  (m) =', med_dv)
            print('[CAL] >>> Вставь R_CORR = ... и DEPTH_SHIFT = ... <<<\n')
        return Rcorr, mean_dv, med_dv

    def clear(self):
        self.Rg.clear(); self.Ro.clear(); self.dv.clear()
