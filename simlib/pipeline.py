# simlib/pipeline.py
from __future__ import annotations

import time
import numpy as np
import mujoco
from spatialmath import SE3

from . import config as cfg
from . import mujoco_io as mj
from . import transforms as tr
from . import sbg_client as sbg
from . import ik_driver as ikd
from . import gripper
from .planner import cartesian_line_plan


def _tcp_dir_down(R_tcp: np.ndarray) -> np.ndarray:
    z = R_tcp[:, 2]
    if np.dot(z, np.array([0, 0, -1])) < 0:
        z = -z
    return z


def _plan_gripper_widths(width_net_m: float) -> tuple[float, float]:
    if not np.isfinite(width_net_m) or width_net_m <= 0.0:
        return float(cfg.FALLBACK_PREOPEN), float(cfg.FALLBACK_CLOSE)
    w_pre = min(width_net_m + float(cfg.PREOPEN_EXTRA_M), float(cfg.GRIPPER_OPEN_M))
    w_close = max(width_net_m - float(cfg.CLOSE_MARGIN_M), float(cfg.GRIPPER_CLOSE_M))
    if w_close > w_pre:
        w_close = w_pre
    return float(w_pre), float(w_close)


def _grasp_succeeded(w_pre: float, w_final: float, f_now: float) -> bool:
    closed_enough = (w_pre - w_final) >= float(cfg.MIN_CLOSE_GAIN_M)
    force_ok = f_now >= float(cfg.SUCCESS_FORCE_N)
    return bool(closed_enough or force_ok)


def _execute_q_path_smooth(ctx: mj.MjContext, q_path, total_time: float = 1.0, viewer=None) -> bool:
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


def _try_side_candidates(ctx: mj.MjContext, G_tcp: SE3, T_from: SE3, viewer=None) -> bool:
    candidates: list[np.ndarray] = []
    base = np.asarray(cfg.SIDE_OFFSET_BASE, float).copy()
    candidates.append(base.copy())
    for dz in cfg.SIDE_Z_DELTAS:
        cand = base.copy(); cand[2] += float(dz)
        candidates.append(cand)
    xy = base[:2]
    for s in cfg.SIDE_XY_SCALES:
        scaled = np.array([xy[0]*float(s), xy[1]*float(s), base[2]], dtype=float)
        candidates.append(scaled.copy())
        for dz in cfg.SIDE_Z_DELTAS:
            cand = scaled.copy(); cand[2] += float(dz)
            candidates.append(cand)

    for i, off in enumerate(candidates):
        T_goal = SE3.Rt(G_tcp.R, T_from.t + off)
        q_side, st = cartesian_line_plan(ctx, T_start=T_from, T_goal=T_goal, steps=int(cfg.SIDE_PLAN_STEPS))
        print(f"[PLAN] retreat→side cand#{i} off={off}  status={st}, N={0 if q_side is None else len(q_side)}")
        if st == "ok" and q_side is not None and len(q_side) > 0:
            return _execute_q_path_smooth(ctx, q_side, total_time=float(cfg.SIDE_MOVE_TIME), viewer=viewer)
    return False


class GraspPipeline:
    def __init__(self, ctx: mj.MjContext, tcp2tip: SE3 | None, sock):
        self.ctx = ctx
        self.tcp2tip = tcp2tip
        self.sock = sock
        self.home_q = ctx.data.qpos.copy()

    def grasp_once(self, pts: np.ndarray, clr: np.ndarray, viewer=None) -> str:
        """
        Выполнить один проход.
        Возвращает:
          "idle_none"  — recv(None), итерация пропущена;
          "idle_empty" — пустой набор grasp’ов;
          "ok"         — цикл выполнен до конца (включая попытку side-move).
        Исключения пробрасываются только для реальных сетевых ошибок send().
        """
        # 2) RPC к SBG
        sbg.send_cloud(self.sock, pts, clr)

        resp = sbg.recv_response(self.sock)
        if resp is None:
            print('[NET] recv: None (timeout/EOF); skip this iteration')
            time.sleep(cfg.SLEEP_BETWEEN)
            return "idle_none"

        # 3) Берём лучший grasp
        grasps = resp.get('grasps', np.zeros((0, 15), np.float32))
        if not isinstance(grasps, np.ndarray):
            grasps = np.asarray(grasps, np.float32)
        if grasps.size == 0:
            time.sleep(cfg.SLEEP_BETWEEN)
            return "idle_empty"

        g_row = grasps[0]
        print(f'RAW grasp row: {g_row}')

        # 4) Парс и ортонорм
        t_cv, R_cv_raw, w, h, d, score, obj_id = sbg.parse_grasp_row(g_row)
        R_cv = tr.ortho_project(R_cv_raw)

        # 5) CV→BASE
        G_net, G_tcp = tr.camcv2base(t_cv, R_cv, mj.T_b_c_gl(self.ctx), depth=d, tcp2tip=self.tcp2tip)
        print('dot(gz_tcp, −Z_world) =', np.dot(G_tcp.R[:,2], [0,0,-1]).round(3))
        print('Целевая TCP с оффсетом:', G_tcp.t)
        print(f'Grasp meta: score={score:.3f}, w={w*1000:.1f}mm, h={h*1000:.1f}mm')

        # 7) Предраскрытие ЗУ
        w_pre, w_close = _plan_gripper_widths(w)
        print(f'[GRIP] pre-open to {w_pre*1000:.1f}mm')
        gripper.gripper_set(self.ctx, w_pre, viewer=viewer)

        # 8) IK в pre-grasp
        z_tcp_down = _tcp_dir_down(G_tcp.R)
        pre_dir_up = -z_tcp_down
        pre_grasp_pos = G_tcp.t + pre_dir_up * float(cfg.PRE_OFFSET_M)
        T_pre = SE3.Rt(G_tcp.R, pre_grasp_pos)

        print(f"[IK] pre-grasp at {pre_grasp_pos}  (dot(pre_dir,+Z_world)={np.dot(pre_dir_up,[0,0,1]):.3f})")
        ok_pre, _ = ikd.goto_arm(self.ctx, T_pre.t, tr.safe_uq_from_R(T_pre.R).vec, viewer=viewer)
        if not ok_pre:
            print('[IK] pre-grasp IK failed')
            time.sleep(cfg.SLEEP_BETWEEN)
            return "ok"  # итерация завершена, но без захвата

        # 9) Выход по прямой
        q_line_down, st2 = cartesian_line_plan(self.ctx, T_start=T_pre, T_goal=G_tcp, steps=80)
        print(f"[PLAN] pre→grasp status={st2}, N={0 if q_line_down is None else len(q_line_down)}")
        if st2 != "ok":
            print("[PLAN] line-plan IK failed; skip grasp and continue")
            time.sleep(cfg.SLEEP_BETWEEN)
            return "ok"

        _execute_q_path_smooth(self.ctx, q_line_down, total_time=1.2, viewer=viewer)

        # 10) Закрытие до силы
        print(f'[GRIP] close until F>={cfg.F_THRESH}N')
        w_final = gripper.gripper_close_until(
            self.ctx,
            f_thresh=float(cfg.F_THRESH),
            min_width=float(w_close),
            timeout_s=2.0,
            viewer=viewer
        )
        print(f'[GRIP] final width={w_final*1000:.1f}mm')

        time.sleep(0.5)

        # 11) Retreat
        T_retreat = SE3.Rt(G_tcp.R, G_tcp.t + pre_dir_up * float(cfg.RET_OFFSET_M))
        q_line_up, st3 = cartesian_line_plan(self.ctx, T_start=G_tcp, T_goal=T_retreat, steps=40)
        print(f"[PLAN] grasp→retreat status={st3}, N={0 if q_line_up is None else len(q_line_up)}")
        if st3 == "ok":
            _execute_q_path_smooth(self.ctx, q_line_up, total_time=1.0, viewer=viewer)
        else:
            ok_ret, _ = ikd.goto_arm(self.ctx, T_retreat.t, tr.safe_uq_from_R(T_retreat.R).vec, viewer=viewer)
            if not ok_ret:
                print("[IK] retreat IK failed; skip post-grasp actions")
                time.sleep(cfg.SLEEP_BETWEEN)
                return "ok"

        # 12) Оценка + уход вбок
        f_now = gripper.gripper_force(self.ctx)
        closed_mm = max(0.0, (w_pre - w_final)) * 1000.0
        ok_grasp = _grasp_succeeded(w_pre, w_final, f_now)
        print(f'[GRIP] eval: ok={ok_grasp}, force={f_now:.1f} N, closed={closed_mm:.1f} mm')

        if ok_grasp:
            try:
                moved = _try_side_candidates(self.ctx, G_tcp, T_from=T_retreat, viewer=viewer)
            except Exception as e:
                print(f"[PLAN] side move failed with error: {e}")
                moved = False

            if moved:
                gripper.gripper_open(self.ctx, viewer=viewer)
                time.sleep(0.3)
            else:
                print("[PLAN] side path failed for all candidates; keep clamped.")
        else:
            gripper.gripper_open(self.ctx, viewer=viewer)
            time.sleep(0.3)

        # 13) Возврат HOME
        print('[IK] going back to HOME...')
        ikd.goto_arm_joints(self.ctx, self.home_q, viewer=viewer)
        time.sleep(cfg.SETTLE_SEC)

        return "ok"
