# simlib/ompl_planner.py

import numpy as np
from ompl import base as ob, geometric as og

def _get_joint_limits_from_mujoco(ctx):
    """Вернуть (low, high) для первых 7 сочленений руки из MuJoCo."""
    low = []
    high = []
    # берём первые 7 шарнирных DOF из qpos (ваш порядок уже согласован в коде)
    for j in range(7):
        # mjModel.jnt_range хранит пределы для каждого сустава (в рад.)
        jr = ctx.model.jnt_range[j]  # shape (2,)
        lo, hi = float(jr[0]), float(jr[1])
        # если сустав "безлимитный" (в MuJoCo бывает jnt_limited=0), задайте дефолт:
        if not ctx.model.jnt_limited[j]:
            lo, hi = -np.pi, np.pi
        # добавим крошечный зазор, чтобы старт/финиш на границах не считались out-of-bounds
        eps = 1e-4
        low.append(lo + eps)
        high.append(hi - eps)
    return np.array(low, dtype=float), np.array(high, dtype=float)


def _make_space_with_bounds(ctx):
    space = ob.RealVectorStateSpace(7)
    low, high = _get_joint_limits_from_mujoco(ctx)
    bounds = ob.RealVectorBounds(7)
    for i in range(7):
        bounds.setLow(i, low[i])
        bounds.setHigh(i, high[i])
    space.setBounds(bounds)
    return space, low, high


def plan_rrt_connect(ctx, q_start, q_goal, time_limit=2.0, range_=0.3):
    """
    Вернуть (path, status): path = список np.array(7,), status=("ok"|"no_path"|"error")
    """
    try:
        space, low, high = _make_space_with_bounds(ctx)
        print("[DBG] bounds low=", low)
        print("[DBG] bounds high=", high)
        print("[DBG] start clipped =", q_s)

        si = ob.SpaceInformation(space)

        # Ваш валидатор состояний (коллизии и т.п.) — оставьте как было:
        def is_state_valid(state):
            q = np.array([state[i] for i in range(7)], dtype=float)
            # TODO: вставьте вашу проверку коллизий MuJoCo, если уже есть
            return True
        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
        si.setStateValidityCheckingResolution(0.005)
        si.setup()

        # Проецируем старт/цель в границы (подстраховка)
        q_s = np.clip(np.asarray(q_start, float).ravel(), low, high)
        q_g = np.clip(np.asarray(q_goal,  float).ravel(), low, high)

        start = ob.State(space)
        goal  = ob.State(space)
        for i in range(7):
            start[i] = float(q_s[i])
            goal[i]  = float(q_g[i])

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)

        planner = og.RRTConnect(si)
        planner.setRange(range_)
        planner.setProblemDefinition(pdef)
        planner.setup()

        solved = planner.solve(time_limit)
        if not solved:
            print("[OMPL] not solved within timelimit")
            return None, "no_path"

        path_geometric = pdef.getSolutionPath()
        path_states = path_geometric.getStates()

        q_path = []
        for st in path_states:
            q = np.array([st[i] for i in range(7)], dtype=float)
            q_path.append(q)
        return q_path, "ok"

    except Exception as e:
        print("[OMPL] exception:", e)
        return None, "error"
