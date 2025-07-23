#!/usr/bin/env python3
"""
run_ik_test.py — офлайн проверка согласованности RTB IK ↔ MuJoCo Panda.

Откроет viewer, поставит несколько тестовых поз (в базе), решит IK,
передвинет суставы, замерит фактическую позу tcp_site и выведет ошибки.

Никакой сети, никакого облака, никакой калибровки.
"""
import time
import numpy as np
import mujoco
import mujoco.viewer
from spatialmath import SE3

from simlib import config as cfg
from simlib import transforms as tr
from simlib import mujoco_io as mj
from simlib import ik_driver as ikd
from simlib import tcp_eval, logutil

def main():
    ctx = mj.load_model(cfg.XML)

    # Показать оси TCP (если настроено)
    with mujoco.viewer.launch_passive(ctx.model, ctx.data) as vw:
        if cfg.SHOW_TCP_AXES_GEOMGROUP >=0:
            vw.opt.geomgroup[cfg.SHOW_TCP_AXES_GEOMGROUP] = 1

        # прогрев
        t0 = time.time()
        while time.time() - t0 < cfg.SETTLE_SEC:
            mujoco.mj_step(ctx.model, ctx.data); vw.sync()

        print("\n--- IK TEST START ---")

        # Несколько учебных целевых поз (вокруг текущего TCP)
        base_tcp0 = mj.tcp_pose_in_base(ctx)  # текущий
        goals = []
        goals.append(base_tcp0)  # текущая
        goals.append(base_tcp0 * SE3.Tz(0.05))  # +5см вверх
        goals.append(base_tcp0 * SE3.Ty(0.05))  # +5см в +Y grasp (в базе не оси grasp; но ок)
        goals.append(SE3.Rx(np.deg2rad(90)) * base_tcp0)  # повернём грубо

        for i,goal in enumerate(goals):
            print(f"\n[Goal {i}]")
            logutil.log_pose("GoalBase", goal)

            ok, qd = ikd.goto_arm(ctx, goal.t, tr.safe_uq_from_R(goal.R).vec, viewer=vw)
            if not ok:
                print("  IK failed; skipping.")
                continue

            # settle
            if cfg.IK_SETTLE_T>0:
                t_set = time.time()+cfg.IK_SETTLE_T
                while time.time()<t_set:
                    mujoco.mj_step(ctx.model, ctx.data); vw.sync()

            # измерить фактический TCP
            act = mj.tcp_pose_in_base(ctx)
            logutil.log_pose("TCPBaseSim", act)

            err = tcp_eval.tcp_error(goal, act)
            print("  err: dP=%.1f мм, dA=%.2f °, gx=%.1f, gy=%.1f, gz=%.1f мм" %
                  (err["dp_mm"], err["da_deg"], err["comp_gx_mm"], err["comp_gy_mm"], err["comp_gz_mm"]))

            time.sleep(0.5)

        print("\n--- IK TEST DONE ---")
        input("Нажми Enter чтобы закрыть viewer...")

if __name__ == "__main__":
    main()
