#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_calib.py — сбор калибровочных данных для ориентации/позиции grasp-сети.

Пайплайн:
  камера → облако → SBG → grasp (в OpenCV-кадре камеры) →
  сравнение с истинной позой эталонного объекта в CV →
  накопление статистики до cfg.CAL_SAMPLES_MAX →
  оценка R_CORR (ориентация) и средних смещений вдоль осей grasp (dv).

Руку не двигаем (IK не нужен).
"""

import time
import cv2
import mujoco, mujoco.viewer
import open3d as o3d
import numpy as np
from pathlib import Path

from simlib import config as cfg
from simlib import mujoco_io as mj
from simlib import mujoco_render as mjr
from simlib import sbg_client as sbg
from simlib import calibration as cal
from simlib import transforms as tr
from simlib.logutil import log_pose, rel_angle_deg, rel_euler_xyz_deg

# локальная задержка цикла, чтобы не шпарить CPU
_LOOP_SLEEP = 0.05


def main():
    # ── MuJoCo / камера / список объектов ──────────────────
    ctx = mj.load_model(cfg.XML)
    cloud = mjr.CloudRenderer(ctx)

    # ── SBG соединение ──────────────────────────────────────
    sock = sbg.connect()

    # ── визуализация облака ─────────────────────────────────
    cv2.namedWindow('RGB')
    vis = o3d.visualization.Visualizer()
    vis.create_window('Cloud', cfg.WIDTH, cfg.HEIGHT)
    pc = o3d.geometry.PointCloud()
    vis.add_geometry(pc)

    # ── аккумулятор калибровочных наблюдений ─────────────────
    acc = cal.CalibAccumulator()
    print_raw = True
    cal_done = False

    with mujoco.viewer.launch_passive(ctx.model, ctx.data) as vw:
        if cfg.SHOW_TCP_AXES_GEOMGROUP >= 0:
            vw.opt.geomgroup[cfg.SHOW_TCP_AXES_GEOMGROUP] = 1

        # прогрев
        t0 = time.time()
        while time.time() - t0 < cfg.SETTLE_SEC:
            mujoco.mj_step(ctx.model, ctx.data)
            vw.sync()

        mj.mj_forward_if_needed(ctx)
        print('\n--- static extrinsics ---')
        TbC = mj.T_b_c_gl(ctx)
        TcB = mj.T_c_b_gl(ctx)
        log_pose('Base->CamGL', TbC)
        log_pose('CamGL->Base', TcB)
        print('-------------------------\n')

        while vw.is_running():
            # 1) обновить облако
            rgb, pts, clr = cloud.grab_cloud_cv()
            cv2.imshow('RGB', rgb)
            cv2.waitKey(1)
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(clr / 255.0)
            vis.update_geometry(pc)
            vis.poll_events()
            vis.update_renderer()

            # 2) отправить облако в SBG
            try:
                sbg.send_cloud(sock, pts, clr)
            except Exception as e:
                print(f'[NET] send error: {e}; reconnect...')
                sock.close()
                sock = sbg.connect()
                continue

            # 3) получить ответ
            resp = sbg.recv_response(sock)
            if resp is None:
                print('[NET] recv failed; reconnect...')
                sock.close()
                sock = sbg.connect()
                continue

            grasps = resp.get('grasps', np.zeros((0, 15), np.float32))
            if not isinstance(grasps, np.ndarray):
                grasps = np.asarray(grasps, np.float32)
            if grasps.size == 0:
                time.sleep(_LOOP_SLEEP)
                continue

            g_row = grasps[0]
            if print_raw:
                print(f'RAW grasp row (len={g_row.shape[0]}): {g_row}')
                print_raw = False

            # 4) разобрать grasp
            t_cam, R_cam_raw, w, h, d, score, obj_id = sbg.parse_grasp_row(g_row)
            R_cam = tr.ortho_project(R_cam_raw)

            # 5) быстрый калибровочный лог и накопление
            if ctx.obj_ids:
                # возьмём ПЕРВЫЙ объект списка как эталон
                To_cv_dbg = mj.obj_in_cam_cv(ctx, ctx.obj_ids[0])  # истинная поза объекта в CV
                ang_dbg = rel_angle_deg(R_cam, To_cv_dbg.R)
                ex, ey, ez = rel_euler_xyz_deg(R_cam, To_cv_dbg.R)
                print(f'[CAL] obj_vs_grasp_cv: angle={ang_dbg:.2f}°  eulerXYZ=({ex:.1f},{ey:.1f},{ez:.1f})')

                # смещение объект - grasp в CV
                dv_xyz = To_cv_dbg.t - t_cam
                gx, gy, gz = R_cam[:, 0], R_cam[:, 1], R_cam[:, 2]
                dvx = np.dot(dv_xyz, gx)
                dvy = np.dot(dv_xyz, gy)
                dvz = np.dot(dv_xyz, gz)
                print(f'[CAL] dv(mm): along_gx={dvx*1000:.1f}  gy={dvy*1000:.1f}  gz={dvz*1000:.1f}')

                # аккумулируем (ИМЕННО компоненты в локальном фрейме grasp!)
                if not cal_done:
                    acc.add(R_cam, To_cv_dbg.R, dv_xyz)
                    if len(acc.Rg) >= cfg.CAL_SAMPLES_MAX:
                        Rcorr, mean_dv, med_dv = acc.done(verbose=True)

                        # сохранить на диск
                        outdir = (cfg.ROOT / 'calib')
                        outdir.mkdir(parents=True, exist_ok=True)
                        np.savez(str(outdir / 'last_calib.npz'),
                                 R_CORR=Rcorr, mean_dv=mean_dv, med_dv=med_dv)
                        cal_done = True

            # 6) печать мета (для наглядности)
            print(f'Grasp meta: score={score:.3f}  w={w*1000:.1f}mm  '
                  f'h={h*1000:.1f}mm  d={d*1000:.1f}mm  obj={obj_id}')
            print()

            time.sleep(_LOOP_SLEEP)

    # ── завершение ───────────────────────────────────────────
    vis.destroy_window()
    cv2.destroyAllWindows()
    cloud.close()


if __name__ == "__main__":
    main()
