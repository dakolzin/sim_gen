"""
sbg_client.py  –  TCP клиент к серверу Scale-Balanced-Grasp (SBG).

Интерфейс:
    sock = connect()
    send_cloud(sock, pts, clr)
    resp = recv_response(sock)   # -> dict {'grasps': np.ndarray(...)}
    t,R,w,h,d,score,obj_id = parse_grasp_row(row)

Зависит от common.py (pack/unpack msgpack + numpy).
"""

import socket
import numpy as np

# импорт сериализации из корневого common.py
from . import common as cmn 

from . import config as cfg

send_msg = cmn.send_msg
recv_msg = cmn.recv_msg
# --------------------------------------------------------------------------- #
# TCP connect
# --------------------------------------------------------------------------- #
def connect():
    """Подключиться к SBG‑серверу; повторять до успеха."""
    while True:
        try:
            s = socket.create_connection((cfg.SBG_HOST, cfg.SBG_PORT))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print('[SIM] connected')
            return s
        except OSError:
            print('[SIM] waiting server...')
            import time
            time.sleep(1)


# --------------------------------------------------------------------------- #
# send / recv
# --------------------------------------------------------------------------- #
def send_cloud(sock: socket.socket, pts: np.ndarray, clr: np.ndarray):
    """
    Отправить облако точек.

    pts : (N,3) float32 (м)
    clr : (N,3) uint8   (RGB)
    """
    send_msg(sock, {'points': pts, 'colors': clr})


def recv_response(sock: socket.socket):
    """
    Принять ответ от сервера. Возвращает dict или None (при разрыве).
    Исключения ConnectionError перекидываем наверх — вызывающая сторона переподключается.
    """
    try:
        return recv_msg(sock)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Grasp row parsing
# --------------------------------------------------------------------------- #
def parse_grasp_row(row: np.ndarray):
    """
    Поддерживаем форматы:
      • (>=15) полный: [score,w,h,d,R(9),t(3),obj_id?]
      • 7: [t(3), quat(4)]
    Возврат: t(3), R(3,3), w,h,d,score,obj_id
    """
    row = np.asarray(row).ravel()
    n = row.shape[0]
    if n >= 15:
        score  = float(row[0])
        width  = float(row[1])
        height = float(row[2])
        depth  = float(row[3])
        R      = row[4:13].reshape(3, 3)
        remain = row[13:]
        if remain.shape[0] >= 4:
            t = remain[:3]
            obj_id = int(remain[3])
        else:
            t = remain[:3]
            obj_id = -1
        return t, R, width, height, depth, score, obj_id

    elif n == 7:
        t = row[:3]
        q = row[3:7]
        # ленивый импорт, чтобы не тянуть spatialmath на старте
        from .transforms import safe_uq_from_q
        R = safe_uq_from_q(q).R
        return t, R, np.nan, np.nan, np.nan, np.nan, -1

    else:
        raise ValueError(f"grasp row len={n} не поддерживается")
