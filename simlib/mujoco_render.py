"""
Рендер RGB-D и построение облака точек в *OpenCV* кадре камеры.
"""
from __future__ import annotations
import numpy as np
import mujoco
from . import config as cfg
from .mujoco_io import MjContext
# Пакет spatialmath не нужен тут.

class CloudRenderer:
    def __init__(self, ctx: MjContext):
        self.ctx = ctx
        self.rgb_r = mujoco.Renderer(ctx.model, height=cfg.HEIGHT, width=cfg.WIDTH)
        self.dep_r = mujoco.Renderer(ctx.model, height=cfg.HEIGHT, width=cfg.WIDTH)
        self.dep_r.enable_depth_rendering()

        # pinhole
        fovy  = np.deg2rad(ctx.model.cam_fovy[ctx.cam_id])
        self.fy = (cfg.HEIGHT/2)/np.tan(fovy/2)
        self.fx = self.fy * cfg.WIDTH / cfg.HEIGHT
        self.cx, self.cy = cfg.WIDTH/2 - .5, cfg.HEIGHT/2 - .5
        u,v = np.meshgrid(np.arange(cfg.WIDTH), np.arange(cfg.HEIGHT))
        self.u = u.ravel(); self.v = v.ravel()

    def grab_cloud_cv(self):
        # RGB
        self.rgb_r.update_scene(self.ctx.data, camera=cfg.CAM_NAME)
        rgb = self.rgb_r.render()[..., ::-1]  # BGR->RGB

        # DEPTH
        self.dep_r.update_scene(self.ctx.data, camera=cfg.CAM_NAME)
        z = self.dep_r.render().ravel()

        m = (z < self.ctx.model.vis.map.zfar) & np.isfinite(z)
        x = (self.u[m]-self.cx) * z[m] / self.fx
        y = (self.v[m]-self.cy) * z[m] / self.fy

        pts = np.c_[x,y,z[m]].astype(np.float32)
        clr = rgb.reshape(-1,3)[m].astype(np.uint8)
        return rgb, pts, clr

    def close(self):
        self.rgb_r.close(); self.dep_r.close()
