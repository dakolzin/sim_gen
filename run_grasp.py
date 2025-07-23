#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_grasp.py — основной цикл:
1. Получение RGB-D облака из камеры
2. Отправка облака в SBG-сервер
3. Выбор лучшего захвата (grasp) из ответа CV-модели
4. Преобразование положения и ориентации grasp из системы координат камеры в систему базы
5. (Опционально) Калибровка и планирование движения через IK Panda (RTB)
6. Оценка и логирование ошибок по положению и ориентации
7. Сценарий захвата: открыть → подъехать → закрыть до силы
8. Пауза и подъём детали на заданное смещение
9. Разжатие ЗУ
"""

import time                     # для пауз и измерения времени
import numpy as np              # численные вычисления
import cv2                      # обработка изображений и вывод окна RGB
import mujoco                   # физический движок MuJoCo
import mujoco.viewer            # просмотр сцены MuJoCo
import open3d as o3d            # визуализация point cloud
from spatialmath import SE3     # работа с преобразованием координат

# Импорт пользовательских модулей из simlib
from simlib import config as cfg       # конфигурация и параметры
from simlib import mujoco_io as mj     # утилиты MuJoCo I/O и матрицы трансформации
from simlib import mujoco_render as mjr# рендеринг облака
from simlib import sbg_client as sbg   # клиент SBG для отправки/приёма облаков
from simlib import transforms as tr    # геометрические преобразования
from simlib import ik_driver as ikd    # драйвер обратной кинематики
from simlib import tcp_eval, logutil   # оценка ошибки TCP и логирование
from simlib import gripper             # управление ЗУ

# --------------- локальные параметры захвата ---------------------------------
_PREOPEN_EXTRA_M   = 0.01   # дополнительное раскрытие перед захватом (м)
_CLOSE_MARGIN_M    = 0.002  # запас перед закрытием (м)
_FALLBACK_PREOPEN  = cfg.GRIPPER_OPEN_M   # запасные значения, если сеть не дала ширину
_FALLBACK_CLOSE    = cfg.GRIPPER_CLOSE_M  #
_F_THRESH          = 150.0  # порог силы для останова закрытия (N)


def _mk_tcp2tip():
    """Вернуть смещение от TCP до кончика инструмента, если включено"""
    if not cfg.USE_TCP_TIP:
        return None
    return SE3.Tz(cfg.TCP2TIP_Z)


def _plan_gripper_widths(width_net_m: float):
    """
    Рассчитать ширину до открытого (pre-open) и до закрытия
    на основе значения, предсказанного CV-сетью.
    """
    if not np.isfinite(width_net_m) or width_net_m <= 0.0:
        return _FALLBACK_PREOPEN, _FALLBACK_CLOSE
    # добавляем маленький зазор к открытому положению
    w_pre = min(width_net_m + _PREOPEN_EXTRA_M, cfg.GRIPPER_OPEN_M)
    # уменьшаем до закрытия с запасом
    w_close = max(width_net_m - _CLOSE_MARGIN_M, cfg.GRIPPER_CLOSE_M)
    if w_close > w_pre:
        w_close = w_pre
    return w_pre, w_close


def main():
    # Загрузка модели MuJoCo и каллибровка инструмента RTB (если нужно)
    ctx = mj.load_model(cfg.XML)
    ikd.calibrate_rtb_tool_from_mj(ctx)

    # Создание рендерера облака и подключение к SBG
    cloud = mjr.CloudRenderer(ctx)
    sock  = sbg.connect()
    tcp2tip = _mk_tcp2tip()

    # Инициализация окон для RGB и облака
    cv2.namedWindow('RGB')
    vis = o3d.visualization.Visualizer()
    vis.create_window('Cloud', cfg.WIDTH, cfg.HEIGHT)
    pc = o3d.geometry.PointCloud()
    vis.add_geometry(pc)

    print_raw = True  # флаг для однократного вывода "сырого" вектора grasp

    with mujoco.viewer.launch_passive(ctx.model, ctx.data) as vw:
        # Опционально включаем оси TCP в визуализации
        if cfg.SHOW_TCP_AXES_GEOMGROUP >= 0:
            vw.opt.geomgroup[cfg.SHOW_TCP_AXES_GEOMGROUP] = 1

        # Ждём установления сцены
        t0 = time.time()
        while time.time() - t0 < cfg.SETTLE_SEC:
            mujoco.mj_step(ctx.model, ctx.data)
            vw.sync()

        # Начальное открытие ЗУ
        print('[GRIP] opening gripper at start...')
        gripper.gripper_open(ctx, viewer=vw)
        time.sleep(cfg.SLEEP_BETWEEN)

        # Логируем статические экстринсики
        mj.mj_forward_if_needed(ctx)
        print('\n--- static extrinsics ---')
        TbC = mj.T_b_c_gl(ctx); logutil.log_pose('Base->CamGL', TbC)
        TcB = mj.T_c_b_gl(ctx); logutil.log_pose('CamGL->Base', TcB)
        print('-------------------------\n')

        # Основной цикл захвата
        while vw.is_running():
            # 1) Считываем RGB и облако точек
            rgb, pts, clr = cloud.grab_cloud_cv()
            cv2.imshow('RGB', rgb); cv2.waitKey(1)
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(clr / 255.0)
            vis.update_geometry(pc); vis.poll_events(); vis.update_renderer()

            # 2) Отправляем облако на сервер и ждём ответ
            try:
                sbg.send_cloud(sock, pts, clr)
            except Exception as e:
                print(f'[NET] send error: {e}; reconnect...')
                sock.close(); sock = sbg.connect(); continue
            resp = sbg.recv_response(sock)
            if resp is None:
                print('[NET] recv failed; reconnect...')
                sock.close(); sock = sbg.connect(); continue

            # 3) Обрабатываем выход сети (список grasp)
            grasps = resp.get('grasps', np.zeros((0, 15), np.float32))
            if not isinstance(grasps, np.ndarray):
                grasps = np.asarray(grasps, np.float32)
            if grasps.size == 0:
                time.sleep(cfg.SLEEP_BETWEEN); continue

            # Берём самый лучший grasp (первый)
            g_row = grasps[0]
            if print_raw:
                print(f'RAW grasp row: {g_row}')
                print_raw = False

            # 4) Парсим параметры grasp и ортонормируем
            t_cv, R_cv_raw, w, h, d, score, obj_id = sbg.parse_grasp_row(g_row)
            R_cv = tr.ortho_project(R_cv_raw)

            # 5) Переводим положение и ориентацию grasp из камеры в базу
            G_net, G_tcp = tr.camcv2base(
                t_cv, R_cv, mj.T_b_c_gl(ctx), depth=d, tcp2tip=None
            )

            # Логируем мета-информацию и положения с ориентацией
            print(f'Grasp meta: score={score:.3f}, w={w*1000:.1f}mm, h={h*1000:.1f}mm')
            logutil.log_pose('GraspBaseNet', G_net)
            logutil.log_pose('GraspBaseTCP', G_tcp)

            # 6) Логируем ошибки по отношению к объектам сцены
            for bid in ctx.obj_ids:
                name = mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                To_b = mj.body_pose_in_base(ctx, bid); logutil.log_pose(f'ObjBase-{name}', To_b)
                To_cv = mj.obj_in_cam_cv(ctx, bid); logutil.log_pose(f'ObjCv-{name}', To_cv)

            # 7) Планируем захват: предварительное раскрытие
            w_pre, w_close = _plan_gripper_widths(w)
            print(f'[GRIP] pre-open to {w_pre*1000:.1f}mm')
            gripper.gripper_set(ctx, w_pre, viewer=vw)

            # Двигаем манипулятор в точку grasp
            ok, _ = ikd.goto_arm(ctx, G_tcp.t, tr.safe_uq_from_R(G_tcp.R).vec, viewer=vw)
            if not ok:
                print('[IK] failed to reach pre-grasp pose'); time.sleep(cfg.SLEEP_BETWEEN); continue

            # Закрытие до силы F_THRESH
            print(f'[GRIP] close until F>={_F_THRESH}N')
            w_final = gripper.gripper_close_until(ctx, f_thresh=_F_THRESH, step_ctrl=5e-4, viewer=vw)
            print(f'[GRIP] final width={w_final*1000:.1f}mm')

            # Пауза перед подъёмом детали
            time.sleep(1)

            # 8) Подъём детали вверх на 0.1м
            lift_offset = np.array([0.0, 0.0, 0.1])
            new_pos = G_tcp.t + lift_offset
            print(f'[GRIP] lifting part by {lift_offset}')
            ok_lift, _ = ikd.goto_arm(ctx, new_pos, tr.safe_uq_from_R(G_tcp.R).vec, viewer=vw)
            if not ok_lift:
                print('[GRIP] lift IK failed')

            # 9) Разжатие ЗУ и возврат к циклу
            gripper.gripper_open(ctx, viewer=vw)
            time.sleep(cfg.SLEEP_BETWEEN)

    # Закрытие окон и освобождение ресурсов
    vis.destroy_window()
    cv2.destroyAllWindows()
    cloud.close()


if __name__ == "__main__":
    main()
