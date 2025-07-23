import numpy as np
from spatialmath import SE3
from simlib import config as cfg
from simlib import transforms as tr

# имитация grasp'а (тест возьми из последнего RAW grasp-а)
row = np.array([ 0.5140805 , 0.04136981, 0.02      , 0.01      ,
                 0.38638002,-0.90155596,-0.19469814, 0.8051152 ,
                 0.43266255,-0.40570018, 0.45      , 0.,
                 0.89302856, 0.01454596,-0.08404332, 0.43063146,-1.])
t_cv, R_cv, w, h, d, s, oid = tr.parse_grasp_row(row)

# поддельные extrinsics base->camGL (из лога)
TbC = SE3( np.array([
    [-0.6533, -0.2706,  0.2706],
    [ 0.2706,  0.6533,  0.2706],
    [-0.2706,  0.2706, -0.6533],
]), [0.100, 0.200, 0.300])

G_net, G_tcp = tr.camcv2base(t_cv, R_cv, TbC, depth=d, tcp2tip=None, debug=True)
print("G_net.t =", G_net.t)
print("G_net.R =\n", G_net.R)
