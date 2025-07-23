# Быстрый тест
import numpy as np
from simlib import transforms as tr, config as cfg
R_cam = np.eye(3)  # возьми тестовую
R_gl = tr.R_CV2GL @ R_cam @ (cfg.R_CORR if cfg.APPLY_R_CORR else np.eye(3))
print(R_gl)
