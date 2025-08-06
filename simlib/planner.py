# simlib/planner.py
# -----------------
# Линейная траектория TCP с покадровым IK (с "семенем" = предыдущее решение).
# Без какой-либо проверки коллизий — только геометрия и IK.
#
# Возврат: (np.ndarray [N×7], "ok") или (None, "ik_fail").

from __future__ import annotations
import numpy as np
from . import transforms as tr
from . import ik_driver as ikd

def cartesian_line_plan(ctx,
                        T_start,
                        T_goal,
                        *,
                        steps: int = 80):
    """
    Построить суставной путь для линейного перемещения TCP от T_start к T_goal.
    Ориентация фиксируется равной ориентации T_goal (без slerp).
    Коллизии НЕ проверяются (упрощённый режим для отладки посадки по вектору).

    Параметры:
      steps – число дискретных точек вдоль отрезка (чем больше, тем плавнее).

    Возврат:
      (np.ndarray [N×7], "ok")  или  (None, "ik_fail")
    """
    p0, p1 = T_start.t, T_goal.t
    R1     = T_goal.R

    qs: list[np.ndarray] = []
    s_grid = np.linspace(0.0, 1.0, int(max(2, steps)))

    for s in s_grid:
        p = (1.0 - s) * p0 + s * p1
        q_xyzw = tr.safe_uq_from_R(R1).vec

        # старт IK = предыдущее решение, иначе текущие суставы
        q0_seed = qs[-1] if qs else ctx.data.qpos[:7].copy()

        # Совместимость: solve_ik может не принимать q0 в старых версиях
        try:
            qd, ok = ikd.solve_ik(p, q_xyzw, q0=q0_seed)
        except TypeError:
            qd, ok = ikd.solve_ik(p, q_xyzw)

        if not ok:
            return None, "ik_fail"

        qs.append(qd)

    return np.asarray(qs, float), "ok"
