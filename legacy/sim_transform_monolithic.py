#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_transform.py – MuJoCo → SBG → (опц.) IK (Franka Panda)

РЕЖИМ КАЛИБРОВКИ (по умолчанию IK отключён, рука не двигается).
Скрипт:
  • Рендерит RGB-D с камеры MuJoCo.
  • Делает back‑projection в облако точек в *OpenCV*-кадре камеры (X→, Y↓, Z→вперёд).
  • Отправляет облако на SBG‑сервер, получает массив grasp'ов формата
    [score,width,height,depth,R(9),t(3),obj_id] (или укороченные варианты).
  • Конвертирует лучший grasp из OpenCV в базовую СК робота: cv→gl (flip Y,Z) → base.
  • Логирует:
       – позу grasp в базе,
       – позу объекта(ов) в базе и в кадре камеры,
       – ошибки grasp↔объект в CV и в базе.
  • Собирает калибровочную статистику ориентаций и позиций (до N примеров) и
    по завершении печатает рекомендуемую матрицу коррекции ориентации R_CORR и
    средний позиционный сдвиг вдоль осей grasp'а.
  • IK Panda можно включить установкой DO_IK=True (тогда требуется Robotics Toolbox).

Минимум воды, весь код целиком (требование пользователя).
"""

import time, socket, pathlib, cv2, mujoco, mujoco.viewer
import open3d as o3d
import numpy as np
from spatialmath import SE3, UnitQuaternion, SO3

# Robotics Toolbox подключаем опционально (чтобы без IK тоже работало)
try:
    import roboticstoolbox as rtb
except Exception:  # не критично в режиме без IK
    rtb = None

from simlib.common import send_msg, unpack   # msgpack helpers

# ───────────────────────── ПАРАМЕТРЫ ─────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[1]
XML       = ROOT/'scene'/'scene.xml'
SBG       = ('127.0.0.1', 6000)
CAM       = 'rs_d435'
H, W      = 480, 640
SETTLE    = 10.0
BASEBODY  = 'link0'

# Смещение от "центра захвата" сети до кончиков пальцев TCP (м):
TCP2TIP   = SE3.Tz(-0.042)   # подправь при необходимости

# --- КАЛИБРОВКА ---
APPLY_R_CORR = True  # применять матрицу ориентационной коррекции

# Матрица коррекции (grasp_cv -> object_cv) по Orthogonal Procrustes (RUN2).
R_CORR = np.array([
    [ 0.91557097,  0.36202273,  0.17512662],
    [ 0.25952523, -0.86453952,  0.43036970],
    [ 0.30720750, -0.34858423, -0.88550132],
], dtype=float)

# Статический офсет grasp-центра в СОБСТВЕННОЙ рамке grasp'а (CV) ДО конверсии.
USE_STATIC_OFFSET = True
G_OFFSET_CV = np.array([0.0487, 0.0142, 0.0060], dtype=float)  # м ~ (gx, gy, gz)

# Доп. глубинная поправка вдоль +Z гриппера (обычно не нужна при статическом офсете).
DEPTH_SHIFT    = 0.0
CAL_SAMPLES_MAX= 25

# --- РЕЖИМЫ ---
DO_IK           = True         # двигаем ли руку
SLEEP_BETWEEN   = 0.05         # задержка цикла
TELEPORT_JOINTS = False        # <<< NEW >>> телепорт суставов вместо PD движения
IK_SETTLE_T     = 0.20         # <<< NEW >>> пауза (сек) после IK перед измерением TCP

# ───────────────────── OpenCV ↔ OpenGL (flip Y,Z) ─────────────────────
# OpenCV: X→, Y↓, Z→вперёд;  OpenGL: X→, Y↑, Z← (в камеру)
R_cv2gl = np.diag([ 1, -1, -1 ])
R_gl2cv = R_cv2gl  # самоинверсная

# ───────────────────── MuJoCo загрузка ─────────────────────
model = mujoco.MjModel.from_xml_path(str(XML))
data  = mujoco.MjData(model)
cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY , BASEBODY)

# <<< NEW >>> site ID TCP (можно None, если site отсутствует)
try:
    TCP_SITE_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
except Exception:
    TCP_SITE_ID = -1

# Renderers
rgb_r = mujoco.Renderer(model, height=H, width=W)
dep_r = mujoco.Renderer(model, height=H, width=W); dep_r.enable_depth_rendering()

# Камерная матрица (fx, fy, cx, cy) из вертикального поля зрения
fovy  = np.deg2rad(model.cam_fovy[cam_id])
fy    = (H/2)/np.tan(fovy/2)
fx    = fy * W / H
cx, cy = W/2 - .5, H/2 - .5
u, v   = np.meshgrid(np.arange(W), np.arange(H)); u=u.ravel(); v=v.ravel()

# ───────────────────── матричные утилиты ─────────────────────
def ortho_project(R):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn

def make_se3(R, t):
    Rn = ortho_project(np.asarray(R, float).reshape(3,3))
    t  = np.asarray(t, float).reshape(3)
    T  = np.eye(4); T[:3,:3]=Rn; T[:3,3]=t
    return SE3(T, check=False)

def safe_uq_from_R(R):
    return UnitQuaternion(SO3(ortho_project(R), check=False))

def safe_uq_from_q(q_xyzw):
    q_xyzw = np.asarray(q_xyzw, float).ravel()
    uq = UnitQuaternion(q_xyzw)
    return UnitQuaternion(SO3(ortho_project(uq.R), check=False))

def rel_angle_deg(Ra, Rb):
    Rrel = Ra @ Rb.T
    c = (np.trace(Rrel) - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def rel_euler_xyz_deg(Ra, Rb):
    Rrel = Ra @ Rb.T
    sy = np.sqrt(Rrel[0,0]**2 + Rrel[1,0]**2)
    if sy < 1e-8:  # сингулярность
        x = np.degrees(np.arctan2(-Rrel[1,2], Rrel[1,1]))
        y = np.degrees(np.arctan2(-Rrel[2,0], sy))
        z = 0.0
    else:
        x = np.degrees(np.arctan2(Rrel[2,1], Rrel[2,2]))
        y = np.degrees(np.arctan2(-Rrel[2,0], sy))
        z = np.degrees(np.arctan2(Rrel[1,0], Rrel[0,0]))
    return x,y,z

def log_pose(tag, T):
    p = T.t; q = safe_uq_from_R(T.R).vec
    print(f"{tag}: pos [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}] quat [{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}]")

# ───────────────────── IK (опц.) ─────────────────────
if rtb is not None:
    panda = rtb.models.Panda()
else:
    panda = None

def goto_arm(pos, quat_xyzw, vw, dur=2.0):
    """
    Решает IK в модели rtb.Panda и двигает суставы MuJoCo.
    Возврат: (success:bool, qd:np.ndarray or None)
    """
    if not DO_IK:
        print('[IK disabled]'); return False, None
    if panda is None:
        print('[IK] Robotics Toolbox недоступен'); return False, None

    uq = safe_uq_from_q(quat_xyzw)
    qd, ok, *_ = panda.ik_LM(SE3.Rt(uq.R, pos))
    if not ok:
        print('[IK] fail'); return False, None

    if TELEPORT_JOINTS:
        # прямое задание позы (для калибровки трансформов)
        data.qpos[:7] = qd
        data.qvel[:7] = 0
        mujoco.mj_forward(model, data)
        vw.sync()
        return True, qd

    # Плавный проход с управлением через ctrl
    q0 = data.ctrl[:7].copy()
    steps = max(2, int(dur / model.opt.timestep))
    for q in np.linspace(q0, qd, steps):
        data.ctrl[:7] = q
        mujoco.mj_step(model, data)
        vw.sync()
        # спим редко, чтобы не замедлять
        time.sleep(model.opt.timestep)
    return True, qd

# ───────────────────── TCP helpers ─────────────────────
def connect():
    while True:
        try:
            s = socket.create_connection(SBG)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print('[SIM] connected'); return s
        except OSError:
            print('[SIM] waiting server...'); time.sleep(1)

sock = connect()

import struct
def recv_msg_safe(sock):
    try:
        hdr = sock.recv(4, socket.MSG_WAITALL)
        if len(hdr) < 4: return None
        sz = struct.unpack('<I', hdr)[0]
        buf = sock.recv(sz, socket.MSG_WAITALL)
        if len(buf) < sz: return None
        return unpack(buf)
    except Exception:
        return None

# ───────────────────── трансформы MuJoCo ─────────────────────
def mj_forward_if_needed():
    mujoco.mj_forward(model, data)

def T_w_b():  # base→world
    mj_forward_if_needed()
    Rwb = np.array(data.xmat[base_id]).reshape(3,3)
    twb = np.array(data.xpos[base_id])
    return make_se3(Rwb, twb)

def T_w_c():  # camGL→world
    mj_forward_if_needed()
    Rwc = np.array(data.cam_xmat[cam_id]).reshape(3,3)
    twc = np.array(data.cam_xpos[cam_id])
    return make_se3(Rwc, twc)

def T_b_c_gl(): return T_w_b().inv() * T_w_c()  # base→camGL
def T_c_b_gl(): return T_b_c_gl().inv()         # camGL→base

# <<< NEW >>> фактическая поза TCP site в базе
def tcp_pose_mj_in_base():
    """Возвращает позу tcp_site в базовой СК (SE3)."""
    if TCP_SITE_ID < 0:
        raise RuntimeError("tcp_site отсутствует в модели.")
    mj_forward_if_needed()
    R_w = np.array(data.site_xmat[TCP_SITE_ID]).reshape(3,3)
    t_w = np.array(data.site_xpos[TCP_SITE_ID])
    return T_w_b().inv() * make_se3(R_w, t_w)

# ───────────────────── парсинг grasp_array ─────────────────────
def parse_grasp_row(row: np.ndarray):
    n = row.shape[0]
    if n >= 15:  # полный формат (17) или без obj_id (15)
        score  = float(row[0]); width  = float(row[1]); height = float(row[2]); depth = float(row[3])
        R      = row[4:13].reshape(3,3)
        remain = row[13:]
        if remain.shape[0] >= 4:
            t = remain[:3]; obj_id = remain[3]
        else:
            t = remain[:3]; obj_id = -1
        return t, R, width, height, depth, score, obj_id
    elif n == 7:  # [t(3), quat(4)]
        t = row[:3]; q = row[3:7]
        R = safe_uq_from_q(q).R
        return t, R, np.nan, np.nan, np.nan, np.nan, -1
    else:
        raise ValueError(f'grasp row len={n} не поддерживается')

# ───────────────────── cv→base ─────────────────────
def camcv2base_tr(t_cv, R_cv, depth=None):
    # cv→gl ориентация
    R_gl = ortho_project(R_cv2gl @ R_cv)
    if APPLY_R_CORR:
        R_gl = R_gl @ R_CORR

    # базовая позиция в gl
    t_gl = R_cv2gl @ t_cv

    # статический офсет в собственном фрейме grasp'а (CV) → GL
    if USE_STATIC_OFFSET:
        t_gl = t_gl + (R_cv2gl @ (R_cv @ G_OFFSET_CV))

    # (опц.) глубинная поправка вдоль оси подхода (+Z гриппера)
    if depth is not None and not np.isnan(depth) and DEPTH_SHIFT != 0.0:
        t_gl = t_gl + R_gl[:,2] * (DEPTH_SHIFT * depth)

    # camGL→grasp
    G_cam = make_se3(R_gl, t_gl)
    # gl→base: baseTgrasp = baseTcam * camTgrasp
    Tbc = T_b_c_gl()
    G_net = Tbc * G_cam
    G_tcp = G_net * TCP2TIP
    return G_net, G_tcp

# ───────────────────── объект в кадре камеры (CV) ─────────────────────
def obj_in_cam_cv(bid):
    R_w_o = np.array(data.xmat[bid]).reshape(3,3)
    t_w_o = np.array(data.xpos[bid])
    T_w_o = make_se3(R_w_o, t_w_o)
    T_w_cgl = T_w_c()
    T_cgl_o = T_w_cgl.inv() * T_w_o   # obj в camGL
    R_c_cv = ortho_project(R_gl2cv @ T_cgl_o.R)
    t_c_cv = R_gl2cv @ T_cgl_o.t
    return make_se3(R_c_cv, t_c_cv)

# ───────────────────── облако (из камеры) ─────────────────────
def grab_cloud_cv():
    rgb_r.update_scene(data, camera=CAM)
    rgb = rgb_r.render()[..., ::-1]  # BGR→RGB
    dep_r.update_scene(data, camera=CAM)
    z = dep_r.render().ravel()       # глубины (м)
    m = (z < model.vis.map.zfar) & np.isfinite(z)
    x = (u[m]-cx) * z[m] / fx        # OpenCV pinhole
    y = (v[m]-cy) * z[m] / fy
    pts = np.c_[x, y, z[m]].astype(np.float32)
    clr = rgb.reshape(-1,3)[m].astype(np.uint8)
    return rgb, pts, clr

# ───────────────────── список объектов для сверки ─────────────────────
obj_ids = [i for i in range(model.nbody)
           if 'fractured' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,i) or '')]

# ───────────────────── визуализация ─────────────────────
cv2.namedWindow('RGB')
vis = o3d.visualization.Visualizer(); vis.create_window('Cloud', W, H)
pc  = o3d.geometry.PointCloud(); vis.add_geometry(pc)

# ───────────────────── калибровочные буферы ─────────────────────
cal_Rg, cal_Ro, cal_dv = [], [], []
cal_done = False

# ───────────────────── главный цикл ─────────────────────
with mujoco.viewer.launch_passive(model, data) as vw:
    vw.opt.geomgroup[4] = 1  # показать TCP оси

    # settle / прогрев
    t0 = time.time()
    while time.time() - t0 < SETTLE:
        mujoco.mj_step(model, data); vw.sync()

    mj_forward_if_needed()
    print('\n--- static extrinsics ---')
    TbC = T_b_c_gl(); TcB = T_c_b_gl()
    log_pose('Base->CamGL', TbC)
    log_pose('CamGL->Base', TcB)
    print('-------------------------\n')

    print_raw = True

    while vw.is_running():
        # обновить кадр + облако
        rgb, pts, clr = grab_cloud_cv()
        cv2.imshow('RGB', rgb); cv2.waitKey(1)
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(clr/255.)
        vis.update_geometry(pc); vis.poll_events(); vis.update_renderer()

        # --- TCP: послать облако ---
        try:
            send_msg(sock, {'points': pts, 'colors': clr})
        except Exception as e:
            print(f'[NET] send error: {e}; reconnect...')
            sock.close(); sock = connect(); continue

        # --- TCP: принять ответ ---
        resp = recv_msg_safe(sock)
        if resp is None:
            print('[NET] recv failed; reconnect...')
            sock.close(); sock = connect(); continue

        grasps = resp.get('grasps', np.zeros((0,15),np.float32))
        if not isinstance(grasps, np.ndarray):
            grasps = np.asarray(grasps, np.float32)
        if grasps.size == 0:
            time.sleep(SLEEP_BETWEEN); continue

        g_row = grasps[0]
        if print_raw:
            print(f'RAW grasp row (len={g_row.shape[0]}): {g_row}')
            print_raw = False

        # разобрать
        t_cam, R_cam_raw, w, h, d, score, obj_id = parse_grasp_row(g_row)
        R_cam = ortho_project(R_cam_raw)

        # --- быстрый калибровочный лог: сопоставление с первым объектом ---
        if obj_ids:
            To_cv_dbg = obj_in_cam_cv(obj_ids[0])
            ang_dbg = rel_angle_deg(R_cam, To_cv_dbg.R)
            ex,ey,ez = rel_euler_xyz_deg(R_cam, To_cv_dbg.R)
            print(f'[CAL] obj_vs_grasp_cv: angle={ang_dbg:.2f}°  eulerXYZ=({ex:.1f},{ey:.1f},{ez:.1f})')
            dv = To_cv_dbg.t - t_cam
            ax = R_cam[:,0]; ay = R_cam[:,1]; az = R_cam[:,2]
            dvx = np.dot(dv, ax); dvy = np.dot(dv, ay); dvz = np.dot(dv, az)
            print(f'[CAL] dv(mm): along_gx={dvx*1000:.1f}  gy={dvy*1000:.1f}  gz={dvz*1000:.1f}')
            if not cal_done:
                cal_Rg.append(R_cam.copy())
                cal_Ro.append(To_cv_dbg.R.copy())
                cal_dv.append([dvx, dvy, dvz])
                if len(cal_Rg) >= CAL_SAMPLES_MAX:
                    A = np.zeros((3,3))
                    for Rg, Ro in zip(cal_Rg, cal_Ro):
                        A += Ro @ Rg.T
                    U,_,Vt = np.linalg.svd(A)
                    R_corr_est = U @ Vt
                    if np.linalg.det(R_corr_est) < 0:
                        U[:,-1] *= -1; R_corr_est = U @ Vt
                    cal_dv_arr = np.array(cal_dv)
                    mean_dv = cal_dv_arr.mean(axis=0)
                    med_dv  = np.median(cal_dv_arr, axis=0)
                    print('\n[CAL] ===== DONE (%d samples) =====' % CAL_SAMPLES_MAX)
                    print('[CAL] R_corr_est =\n', R_corr_est)
                    print('[CAL] det=%.6f' % np.linalg.det(R_corr_est))
                    print('[CAL] mean_dv_components (m) =', mean_dv)
                    print('[CAL] med_dv_components  (m) =', med_dv)
                    print('[CAL] >>> Вставь R_CORR = ... и DEPTH_SHIFT = ... <<<\n')
                    cal_done = True

        # cv→base
        G_net, G_tcp = camcv2base_tr(t_cam, R_cam, d)

        # логи захвата
        print(f'Grasp meta: score={score:.3f}  w={w*1000:.1f}mm  h={h*1000:.1f}mm  d={d*1000:.1f}mm  obj={obj_id}')
        log_pose('GraspBaseNet', G_net)
        log_pose('GraspBaseTCP', G_tcp)

        # диагностика по всем объектам
        T_wb = T_w_b(); T_bw = T_wb.inv()
        for bid in obj_ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            # объект в базе
            R_w_o = np.array(data.xmat[bid]).reshape(3,3)
            t_w_o = np.array(data.xpos[bid])
            T_w_o = make_se3(R_w_o, t_w_o)
            To_b  = T_bw * T_w_o
            log_pose(f'ObjBase-{name}', To_b)
            # объект в CV
            To_cv = obj_in_cam_cv(bid)
            log_pose(f'ObjCv-{name}', To_cv)
            # ошибки
            p_err_cv = np.linalg.norm(t_cam - To_cv.t) * 1000.0
            a_err_cv = rel_angle_deg(R_cam, To_cv.R)
            print(f'Δ_cv grasp-obj({name}): {p_err_cv:.1f} мм  {a_err_cv:.2f} °')
            p_err_b  = np.linalg.norm(G_net.t - To_b.t) * 1000.0
            a_err_b  = rel_angle_deg(G_net.R, To_b.R)
            print(f'Δ_base grasp-obj({name}): {p_err_b:.1f} мм  {a_err_b:.2f} °')

        # ───── IK / движение / измерение фактического TCP ─────
        if DO_IK:
            ok, qd = goto_arm(G_tcp.t, safe_uq_from_R(G_tcp.R).vec, vw)
            if ok:
                # дать время системе устояться, если не телепорт
                if not TELEPORT_JOINTS and IK_SETTLE_T > 0:
                    t_set = time.time() + IK_SETTLE_T
                    while time.time() < t_set:
                        mujoco.mj_step(model, data); vw.sync()
                        time.sleep(model.opt.timestep)

                # поза rtb.Panda по текущим qpos (диагн. рассинхрон кинематик)
                if panda is not None:
                    ee_base_rtb = panda.fkine(data.qpos[:7])
                    log_pose('EEBase_rtb', ee_base_rtb)
                    dp_rtb = np.linalg.norm(ee_base_rtb.t - G_tcp.t) * 1000.0
                    da_rtb = rel_angle_deg(ee_base_rtb.R, G_tcp.R)
                    print(f'err(rtb TCP-goal): {dp_rtb:.1f} мм  {da_rtb:.2f} °')

                # фактическая поза TCP из MuJoCo
                try:
                    TCPBaseSim = tcp_pose_mj_in_base()
                    log_pose('TCPBaseSim', TCPBaseSim)
                    dp_sim = np.linalg.norm(TCPBaseSim.t - G_tcp.t) * 1000.0
                    da_sim = rel_angle_deg(TCPBaseSim.R, G_tcp.R)
                    print(f'err(sim TCP-goal): {dp_sim:.1f} мм  {da_sim:.2f} °')
                    # декомпозиция по осям целевого grasp'а
                    dv = TCPBaseSim.t - G_tcp.t
                    gx, gy, gz = G_tcp.R[:,0], G_tcp.R[:,1], G_tcp.R[:,2]
                    print(f'   components: gx={np.dot(dv,gx)*1000:.1f}  gy={np.dot(dv,gy)*1000:.1f}  gz={np.dot(dv,gz)*1000:.1f} мм')
                except RuntimeError as e:
                    print('[TCP meas] %s' % e)
            print()  # пустая строка
        else:
            print()  # пустая строка для читаемости

        time.sleep(SLEEP_BETWEEN)

# ───────────────────── завершение ─────────────────────
vis.destroy_window(); cv2.destroyAllWindows()
rgb_r.close(); dep_r.close()
