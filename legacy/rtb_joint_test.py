#!/usr/bin/env python3
import numpy as np
import roboticstoolbox as rtb

# Загрузим URDF Panda
p = rtb.models.URDF.Panda()

# Конфигурация: нули (можешь заменить p.qr и т.п.)
q = np.zeros(p.n)   # p.n == 7

# Якобиан в базе; 6×n: [vx vy vz wx wy wz]^T
J = p.jacob0(q)

print("RTB Panda joint axes in BASE frame (at q=zeros):")
for i in range(p.n):
    w = J[3:6, i]
    if np.linalg.norm(w) < 1e-9:
        print(f"joint{i+1}: prismatic? (w≈0)")
    else:
        w_n = w / np.linalg.norm(w)
        print(f"joint{i+1}: axis={w_n}")
