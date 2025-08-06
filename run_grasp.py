#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import time, socket
import cv2, mujoco, mujoco.viewer, open3d as o3d
from spatialmath import SE3

from simlib import config as cfg, mujoco_io as mj, mujoco_render as mjr, sbg_client as sbg
from simlib.pipeline import GraspPipeline
from simlib import gripper

def _mk_tcp2tip():
    if not cfg.USE_TCP_TIP:
        return None
    return SE3.Tz(cfg.TCP2TIP_Z)

def main():
    ctx = mj.load_model(cfg.XML)

    from simlib import ik_driver as ikd
    ikd.calibrate_rtb_tool_from_mj(ctx)

    cloud = mjr.CloudRenderer(ctx)
    sock  = sbg.connect()
    tcp2tip = _mk_tcp2tip()
    print("DEBUG: tcp2tip =", tcp2tip)

    cv2.namedWindow('RGB')
    vis = o3d.visualization.Visualizer()
    vis.create_window('Cloud', cfg.WIDTH, cfg.HEIGHT)
    pc = o3d.geometry.PointCloud(); vis.add_geometry(pc)

    pipe = GraspPipeline(ctx, tcp2tip=tcp2tip, sock=sock)

    with mujoco.viewer.launch_passive(ctx.model, ctx.data) as vw:
        if cfg.SHOW_TCP_AXES_GEOMGROUP >= 0:
            vw.opt.geomgroup[cfg.SHOW_TCP_AXES_GEOMGROUP] = 1

        t0 = time.time()
        while time.time() - t0 < cfg.SETTLE_SEC:
            mujoco.mj_step(ctx.model, ctx.data); vw.sync()

        print('[GRIP] opening gripper at start...')
        gripper.gripper_open(ctx, viewer=vw); time.sleep(cfg.SLEEP_BETWEEN)

        mj.mj_forward_if_needed(ctx)
        print('\n--- static extrinsics ---')
        print('Base->CamGL:', mj.T_b_c_gl(ctx))
        print('CamGL->Base:', mj.T_c_b_gl(ctx))
        print('-------------------------\n')

        net_fail_cnt = 0
        idle_cnt = 0  # подряд "пустых" итераций (recv None или 0 grasps)

        while vw.is_running():
            # 1) RGB + облако
            rgb, pts, clr = cloud.grab_cloud_cv()
            cv2.imshow('RGB', rgb); cv2.waitKey(1)
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(clr / 255.0)
            vis.update_geometry(pc); vis.poll_events(); vis.update_renderer()

            try:
                status = pipe.grasp_once(pts, clr, viewer=vw)
            except (socket.error, ConnectionError, TimeoutError, OSError) as e:
                net_fail_cnt += 1
                print(f'[NET] error: {e}; reconnect...  ({net_fail_cnt}/{cfg.NET_FAIL_LIMIT})')
                if net_fail_cnt >= cfg.NET_FAIL_LIMIT:
                    print('[NET] превышен лимит неудачных операций — выходим.')
                    break
                try:
                    sock.close()
                except Exception:
                    pass
                sock = sbg.connect()
                pipe.sock = sock
                continue
            else:
                net_fail_cnt = 0  # успех сети

            # Учёт "пустых" итераций
            if status in ("idle_none", "idle_empty"):
                idle_cnt += 1
                print(f"[IDLE] {status}  ({idle_cnt}/{cfg.IDLE_LIMIT})")
                if idle_cnt >= cfg.IDLE_LIMIT:
                    print("[IDLE] достигнут предел пустых итераций — завершаем работу.")
                    break
            else:
                idle_cnt = 0  # была полноценная попытка

    vis.destroy_window(); cv2.destroyAllWindows(); cloud.close()

if __name__ == "__main__":
    main()
