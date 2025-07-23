"""
simlib – внутренние утилиты симулятора D435 + Panda.

Смотри отдельные подмодули:
  config          – пути, параметры калибровки, флаги режимов
  transforms      – матр. утилиты, cv↔gl, cv→base (с коррекциями)
  mujoco_io       – загрузка модели, доступ к позам, extrinsics, TCP site
  mujoco_render   – RGB-D рендер, back-projection в облако
  sbg_client      – TCP клиент, отправка облака, парсинг grasp'ов
  ik_driver       – RTB IK → MuJoCo движение (interp/teleport)
  calibration     – накопление и оценка R_CORR, dv
  tcp_eval        – сравнение целевого TCP и фактического из MuJoCo
  logutil         – печать поз в человекочитаемом формате
"""
from . import config, transforms, mujoco_io, mujoco_render, sbg_client, ik_driver, calibration, tcp_eval, logutil, common, gripper
__all__ = ["config","transforms","mujoco_io","mujoco_render","sbg_client","ik_driver","calibration","tcp_eval","logutil", "common", "gripper"]
