#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_grasp.py — основной цикл:
1) RGB-D облако → SBG
2) Берём лучший grasp
3) CV→BASE преобразование
4) Движение: IK к pre-grasp → посадка по прямой (без коллизий)
5) Закрытие до силы
6) Retreat по прямой вверх
7) Если захват успешен — сдвиг в сторону (с перебором кандидатов) и раскрытие;
   иначе — раскрытие на retreat
8) Возврат HOME
"""

from __future__ import annotations
import time
import numpy as np
import cv2
import mujoco
import mujoco.viewer
import open3d as o3d
from spatialmath import SE3

# simlib
from simlib import config as cfg
from simlib import mujoco_io as mj
from simlib import mujoco_render as mjr
from simlib import sbg_client as sbg
from simlib import transforms as tr
from simlib import ik_driver as ikd
from simlib import tcp_eval, logutil
from simlib import gripper
from simlib.planner import cartesian_line_plan

# ---------------- параметры ----------------
_PREOPEN_EXTRA_M   = 0.01
_CLOSE_MARGIN_M    = 0.002
_FALLBACK_PREOPEN  = cfg.GRIPPER_OPEN_M
_FALLBACK_CLOSE    = cfg.GRIPPER_CLOSE_M
_F_THRESH          = 150.0

_PRE_OFFSET_M  = 0.15   # pre-grasp на 15 см «вверх от детали»
_RET_OFFSET_M  = 0.12   # retreat на 12 см «вверх от детали»

# Успех захвата / уход в сторону
_SUCCESS_FORCE_N     = 40.0
_MIN_CLOSE_GAIN_M    = 0.006

# Базовое смещение «в сторону» в БАЗОВОЙ СК
_SIDE_OFFSET_BASE    = np.array([-0.20, -0.30, 0.00])  # м

# Перебор альтернативных боковых точек:
_SIDE_Z_DELTAS       = [0.10, -0.10, 0.18, -0.18]      # добавить к Z (м)
_SIDE_XY_SCALES      = [1.0, 1.3, 0.7]                 # множители для XY-смещения
_SIDE_PLAN_STEPS     = 40
_SIDE_MOVE_TIME      = 1.0

# Авто-отключение при сетевых проблемах
_NET_FAIL_LIMIT      = 5  # <<— если 5 подряд неудач — выходим

# -------------- утилиты --------------------
def _grasp_succeeded(w_pre: float, w_final: float, f_now: float) -> bool:
    """Эвристика успеха захвата: достаточно закрылись или сила ≥ порога."""
    closed_enough = (w_pre - w_final) >= _MIN_CLOSE_GAIN_M
    force_ok = f_now >= _SUCCESS_FORCE_N
    return bool(closed_enough or force_ok)

def _mk_tcp2tip():
    if not cfg.USE_TCP_TIP:
        return None
    return SE3.Tz(cfg.TCP2TIP_Z)

def _plan_gripper_widths(width_net_m: float):
    """Определить предраскрытие и целевое закрытие из предсказанной ширины."""
    if not np.isfinite(width_net_m) or width_net_m <= 0.0:
        return _FALLBACK_PREOPEN, _FALLBACK_CLOSE
    w_pre = min(width_net_m + _PREOPEN_EXTRA_M, cfg.GRIPPER_OPEN_M)
    w_close = max(width_net_m - _CLOSE_MARGIN_M, cfg.GRIPPER_CLOSE_M)
    if w_close > w_pre:
        w_close = w_pre
    return w_pre, w_close

def _tcp_dir_down(R_tcp: np.ndarray) -> np.ndarray:
    """Вернуть направление -Z_мiра ближайшее к оси z TCP (вниз к столу)."""
    z = R_tcp[:, 2]
    if np.dot(z, np.array([0, 0, -1])) < 0:
        z = -z
    return z

def _print_after_move_logs(ctx, G_tcp: SE3):
    TCP = mj.tcp_pose_in_base(ctx)
    try:
        FLANGE = ikd._PANDA.fkine(ctx.data.qpos[:7]).t
    except Exception:
        FLANGE = TCP.t
    print('┌── after move')
    print('│ goal TCP Z   =', round(G_tcp.t[2]*1000,1), 'мм')
    print('│ fact TCP Z   =', round(TCP.t[2]*1000,1),   'мм')
    print('│ fact FLANGE Z=', round(FLANGE[2]*1000,1),  'мм')
    print('│ ΔZ flange-vs-goal =', round((FLANGE[2]-G_tcp.t[2])*1000,1),'мм')
    print('└──────────────')

    TCPBaseSim = mj.tcp_pose_in_base(ctx)
    err = tcp_eval.tcp_error(G_tcp, TCPBaseSim)
    print('Фактическая TCP в симуляции:', TCPBaseSim.t)
    print(f'Δpos = {err["dp_mm"]:.1f} мм, Δang = {err["da_deg"]:.2f}°')

def execute_q_path_smooth(ctx, q_path, total_time=1.0, viewer=None):
    """Плавно пройти по последовательности суставных векторов q_path за total_time."""
    import numpy as np
    q_path = np.asarray(q_path, float)
    if q_path.ndim == 1:
        q_path = q_path.reshape(1, -1)

    n = q_path.shape[0]
    if n == 0:
        return False
    q0 = ctx.data.qpos[:q_path.shape[1]].copy()
    knots = np.vstack([q0, q_path])

    total_steps = max(200, int(total_time / ctx.model.opt.timestep))
    s_grid = np.linspace(0.0, 1.0, total_steps)

    seg_len = np.linalg.norm(np.diff(knots, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    if cum[-1] < 1e-9:
        ctx.data.ctrl[:q_path.shape[1]] = knots[-1]
        mujoco.mj_forward(ctx.model, ctx.data)
        if viewer:
            viewer.sync()
        return True

    u = cum / cum[-1]
    from numpy import interp
    for s in s_grid:
        q = np.array([interp(s, u, knots[:, j]) for j in range(knots.shape[1])])
        ctx.data.ctrl[:q.size] = q
        mujoco.mj_step(ctx.model, ctx.data)
        if viewer:
            viewer.sync()
    return True

def _try_side_candidates(ctx, G_tcp: SE3, T_from: SE3, viewer=None) -> bool:
    """
    Перебор кандидатов для 'ухода вбок': базовое смещение, затем те же XY,
    но выше/ниже (±Z_DELTAS), затем — с масштабом XY (±Z тоже).
    Успех: траектория построена и выполнена → True, иначе False.
    """
    candidates = []

    # 1) Базовая точка
    base = _SIDE_OFFSET_BASE.copy()
    candidates.append(base.copy())

    # 2) Та же сторона выше/ниже
    for dz in _SIDE_Z_DELTAS:
        cand = base.copy()
        cand[2] += dz
        candidates.append(cand)

    # 3) Масштаб по XY (+ вариации по Z)
    xy = base[:2]
    for s in _SIDE_XY_SCALES:
        scaled = np.array([xy[0]*s, xy[1]*s, base[2]], dtype=float)
        candidates.append(scaled.copy())
        for dz in _SIDE_Z_DELTAS:
            cand = scaled.copy()
            cand[2] += dz
            candidates.append(cand)

    # Пробуем по очереди
    for i, off in enumerate(candidates):
        T_goal = SE3.Rt(G_tcp.R, T_from.t + off)
        q_side, st = cartesian_line_plan(ctx, T_start=T_from, T_goal=T_goal, steps=_SIDE_PLAN_STEPS)
        print(f"[PLAN] retreat→side cand#{i} off={off}  status={st}, N={0 if q_side is None else len(q_side)}")
        if st == "ok" and q_side is not None and len(q_side) > 0:
            return execute_q_path_smooth(ctx, q_side, total_time=_SIDE_MOVE_TIME, viewer=viewer)
    return False

# -------------- основной цикл ----------------
def main():
    ctx = mj.load_model(cfg.XML)
    HOME_Q = ctx.data.qpos.copy()
    ikd.calibrate_rtb_tool_from_mj(ctx)

    cloud = mjr.CloudRenderer(ctx)
    sock  = sbg.connect()
    tcp2tip = _mk_tcp2tip()
    print("DEBUG: tcp2tip =", tcp2tip)

    cv2.namedWindow('RGB')
    vis = o3d.visualization.Visualizer()
    vis.create_window('Cloud', cfg.WIDTH, cfg.HEIGHT)
    pc = o3d.geometry.PointCloud()
    vis.add_geometry(pc)

    print_raw = True

    with mujoco.viewer.launch_passive(ctx.model, ctx.data) as vw:
        if cfg.SHOW_TCP_AXES_GEOMGROUP >= 0:
            vw.opt.geomgroup[cfg.SHOW_TCP_AXES_GEOMGROUP] = 1

        t0 = time.time()
        while time.time() - t0 < cfg.SETTLE_SEC:
            mujoco.mj_step(ctx.model, ctx.data)
            vw.sync()

        print('[GRIP] opening gripper at start...')
        gripper.gripper_open(ctx, viewer=vw)
        time.sleep(cfg.SLEEP_BETWEEN)

        mj.mj_forward_if_needed(ctx)
        print('\n--- static extrinsics ---')
        TbC = mj.T_b_c_gl(ctx); logutil.log_pose('Base->CamGL', TbC)
        TcB = mj.T_c_b_gl(ctx); logutil.log_pose('CamGL->Base', TcB)
        print('-------------------------\n')

        # --- сетевой счётчик неудач ---
        net_fail_cnt = 0
        stop_due_to_net = False

        while vw.is_running():
            if stop_due_to_net:
                break

            # 1) RGB + облако
            rgb, pts, clr = cloud.grab_cloud_cv()
            cv2.imshow('RGB', rgb); cv2.waitKey(1)
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(clr / 255.0)
            vis.update_geometry(pc); vis.poll_events(); vis.update_renderer()

            # 2) RPC к SBG
            try:
                sbg.send_cloud(sock, pts, clr)
            except Exception as e:
                net_fail_cnt += 1
                print(f'[NET] send error: {e}; reconnect...  ({net_fail_cnt}/{_NET_FAIL_LIMIT})')
                if net_fail_cnt >= _NET_FAIL_LIMIT:
                    print('[NET] превышен лимит неудачных отправок — выходим из программы.')
                    stop_due_to_net = True
                    break
                try:
                    sock.close()
                except Exception:
                    pass
                sock = sbg.connect()
                continue

            resp = sbg.recv_response(sock)
            if resp is None:
                net_fail_cnt += 1
                print(f'[NET] recv failed; reconnect...  ({net_fail_cnt}/{_NET_FAIL_LIMIT})')
                if net_fail_cnt >= _NET_FAIL_LIMIT:
                    print('[NET] превышен лимит неудачных приёмов — выходим из программы.')
                    stop_due_to_net = True
                    break
                try:
                    sock.close()
                except Exception:
                    pass
                sock = sbg.connect()
                continue
            else:
                # получен ответ — сбрасываем счётчик неудач
                net_fail_cnt = 0

            # 3) Берём лучший grasp
            grasps = resp.get('grasps', np.zeros((0, 15), np.float32))
            if not isinstance(grasps, np.ndarray):
                grasps = np.asarray(grasps, np.float32)
            if grasps.size == 0:
                time.sleep(cfg.SLEEP_BETWEEN); continue

            g_row = grasps[0]
            if print_raw:
                print(f'RAW grasp row: {g_row}')
                print_raw = False

            # 4) Парс и ортонорм
            t_cv, R_cv_raw, w, h, d, score, obj_id = sbg.parse_grasp_row(g_row)
            R_cv = tr.ortho_project(R_cv_raw)

            # 5) CV→BASE
            G_net, G_tcp = tr.camcv2base(t_cv, R_cv, mj.T_b_c_gl(ctx), depth=d, tcp2tip=tcp2tip)
            print('dot(gz_tcp, −Z_world) =', np.dot(G_tcp.R[:,2], [0,0,-1]).round(3))
            print('Целевая TCP с оффсетом:', G_tcp.t)

            print(f'Grasp meta: score={score:.3f}, w={w*1000:.1f}mm, h={h*1000:.1f}mm')
            logutil.log_pose('GraspBaseNet', G_net)
            logutil.log_pose('GraspBaseTCP', G_tcp)

            # 6) (опц.) лог поз объектов
            for bid in ctx.obj_ids:
                name = mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                To_b = mj.body_pose_in_base(ctx, bid); logutil.log_pose(f'ObjBase-{name}', To_b)
                To_cv = mj.obj_in_cam_cv(ctx, bid);   logutil.log_pose(f'ObjCv-{name}', To_cv)

            # 7) Предраскрытие ЗУ
            w_pre, w_close = _plan_gripper_widths(w)
            print(f'[GRIP] pre-open to {w_pre*1000:.1f}mm')
            gripper.gripper_set(ctx, w_pre, viewer=vw)

            # 8) IK в pre-grasp (на безопасной высоте)
            z_tcp_down = _tcp_dir_down(G_tcp.R)
            pre_dir_up = -z_tcp_down
            pre_grasp_pos = G_tcp.t + pre_dir_up * _PRE_OFFSET_M
            T_pre = SE3.Rt(G_tcp.R, pre_grasp_pos)

            print(f"[IK] pre-grasp at {pre_grasp_pos}  (dot(pre_dir,+Z_world)={np.dot(pre_dir_up,[0,0,1]):.3f})")
            ok_pre, _ = ikd.goto_arm(ctx, T_pre.t, tr.safe_uq_from_R(T_pre.R).vec, viewer=vw)
            if not ok_pre:
                print('[IK] pre-grasp IK failed')
                time.sleep(cfg.SLEEP_BETWEEN)
                continue

            # 9) Посадка по прямой: pre → grasp (БЕЗ коллизий)
            q_line_down, st2 = cartesian_line_plan(ctx, T_start=T_pre, T_goal=G_tcp, steps=80)
            print(f"[PLAN] pre→grasp status={st2}, N={0 if q_line_down is None else len(q_line_down)}")
            if st2 != "ok":
                print("[PLAN] line-plan IK failed; skip grasp and continue")
                time.sleep(cfg.SLEEP_BETWEEN)
                continue

            execute_q_path_smooth(ctx, q_line_down, total_time=1.2, viewer=vw)

            # Логи после прихода в grasp
            _print_after_move_logs(ctx, G_tcp)
            print(">>> APPLY_R_CORR=", cfg.APPLY_R_CORR, "USE_STATIC_OFFSET=", cfg.USE_STATIC_OFFSET)
            print(">>> R_CORR=\n", cfg.R_CORR)
            print(">>> G_OFFSET_CV=", cfg.G_OFFSET_CV, "DEPTH_SHIFT=", cfg.DEPTH_SHIFT)

            # 10) Закрытие до силы
            print(f'[GRIP] close until F>={_F_THRESH}N')
            w_final = gripper.gripper_close_until(
                ctx,
                f_thresh=_F_THRESH,
                min_width=None,           # можно поставить w_close, если нужен «стоп по ширине»
                timeout_s=2.0,
                viewer=vw
            )
            print(f'[GRIP] final width={w_final*1000:.1f}mm')

            time.sleep(0.5)

            # 11) Retreat по прямой вверх
            T_retreat = SE3.Rt(G_tcp.R, G_tcp.t + pre_dir_up * _RET_OFFSET_M)
            q_line_up, st3 = cartesian_line_plan(ctx, T_start=G_tcp, T_goal=T_retreat, steps=40)
            print(f"[PLAN] grasp→retreat status={st3}, N={0 if q_line_up is None else len(q_line_up)}")
            if st3 == "ok":
                execute_q_path_smooth(ctx, q_line_up, total_time=1.0, viewer=vw)
            else:
                ok_ret, _ = ikd.goto_arm(ctx, T_retreat.t, tr.safe_uq_from_R(T_retreat.R).vec, viewer=vw)
                if not ok_ret:
                    print("[IK] retreat IK failed; skip post-grasp actions")
                    time.sleep(cfg.SLEEP_BETWEEN)
                    continue

            # 12) Оценка успеха захвата и «уход вбок»
            f_now = gripper.gripper_force(ctx)
            ok_grasp = _grasp_succeeded(w_pre, w_final, f_now)
            print(f'[GRIP] eval: ok={ok_grasp}, force={f_now:.1f} N, closed={(w_pre - w_final)*1000:.1f} mm')

            if ok_grasp:
                # Пытаемся уехать вбок (с вариациями по Z и масштабу XY)
                moved = _try_side_candidates(ctx, G_tcp, T_from=T_retreat, viewer=vw)
                if moved:
                    # Только если уехали — отпускаем
                    gripper.gripper_open(ctx, viewer=vw)
                    time.sleep(0.3)
                else:
                    print("[PLAN] side path failed for all candidates; keep clamped.")
                    # Можно добавить: уйти выше и повторить _try_side_candidates
            else:
                gripper.gripper_open(ctx, viewer=vw)
                time.sleep(0.3)

            # 13) Возврат HOME
            print('[IK] going back to HOME...')
            ikd.goto_arm_joints(ctx, HOME_Q, viewer=vw)
            time.sleep(cfg.SETTLE_SEC)

    vis.destroy_window()
    cv2.destroyAllWindows()
    cloud.close()

if __name__ == "__main__":
    main()
