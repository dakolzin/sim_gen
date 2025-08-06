#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mjcf2urdf_panda.py   –   MJCF → URDF (RTB/ROS), 7 DOF + фиксированные пальцы
---------------------------------------------------------------------------
usage:
    python mjcf2urdf_panda.py  scene/panda_fixed.xml  panda_from_mjcf.urdf
    python mjcf2urdf_panda.py scene/panda_fixed.xml panda_from_mjcf.urdf
"""

import sys, math, xml.etree.ElementTree as ET
from pathlib import Path

# ---------------- helpers ----------------------------------------------------
def vec(s: str, pad=0):
    out = list(map(float, s.strip().split()))
    return out + [0.0] * max(0, pad - len(out))

def quat_wxyz_to_rpy(q):
    w,x,y,z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin (max(-1, min(1, 2*(w*y - z*x))))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw

def indent(e, lvl=0):
    i = "\n" + "  " * lvl
    if len(e):
        e.text = i + "  "
        for c in e:
            indent(c, lvl+1)
            c.tail = i
    if lvl and (e.tail is None or not e.tail.strip()):
        e.tail = i

# ---------------- args -------------------------------------------------------
if len(sys.argv) != 3:
    sys.exit("usage: mjcf2urdf_panda.py  in.xml  out.urdf")
in_path, out_path = map(Path, sys.argv[1:])

root = ET.parse(in_path).getroot()
mesh_map = {m.get('name'): m.get('file') for m in root.findall('./asset/mesh')}

links, joints = {}, []

# ---------------- recursion --------------------------------------------------
def process_body(body, parent):
    name = body.get('name')

    pos_b  = vec(body.get('pos',  '0 0 0'), pad=3)
    quat_b = vec(body.get('quat', '0 0 0 1'), pad=4)
    rpy_b  = quat_wxyz_to_rpy([quat_b[3], *quat_b[:3]])

    # ------------- link ------------------------------------------------------
    link_xml = ET.Element('link', {'name': name})

    inert = body.find('inertial')
    if inert is not None and inert.get('mass'):
        mass = inert.get('mass')
        diag = vec(inert.get('diaginertia', '0 0 0'), pad=3)
        full = vec(inert.get('fullinertia',  ' '.join(map(str, diag))), pad=6)
        com  = vec(inert.get('pos', '0 0 0'), pad=3)
        ix = ET.SubElement(link_xml, 'inertial')
        ET.SubElement(ix, 'origin',
                      {'xyz': f"{com[0]} {com[1]} {com[2]}", 'rpy': "0 0 0"})
        ET.SubElement(ix, 'mass', {'value': mass})
        ET.SubElement(ix, 'inertia',
                      {'ixx': str(full[0]), 'iyy': str(full[1]), 'izz': str(full[2]),
                       'ixy': str(full[3]), 'ixz': str(full[4]), 'iyz': str(full[5])})

    for g in body.findall('geom'):
        mesh = g.get('mesh');  file = mesh_map.get(mesh, mesh)
        if not file: continue
        pos  = vec(g.get('pos',  '0 0 0'), pad=3)
        quat = vec(g.get('quat', '0 0 0 1'), pad=4)
        rpy  = quat_wxyz_to_rpy([quat[3], *quat[:3]])
        for tag in ('visual', 'collision'):
            blk = ET.SubElement(link_xml, tag)
            ET.SubElement(blk, 'origin',
                          {'xyz': f"{pos[0]} {pos[1]} {pos[2]}",
                           'rpy': f"{rpy[0]} {rpy[1]} {rpy[2]}"})
            geom = ET.SubElement(blk, 'geometry')
            ET.SubElement(geom, 'mesh', {'filename': file})

    links[name] = link_xml

    # ------------- joint -----------------------------------------------------
    mj_j = body.find('joint')
    if parent is not None:
        # --- решаем тип: пальцы и без‑joint → fixed -------------------------
        if mj_j is None or name.startswith(('left_finger', 'right_finger', 'hand', 'tcp')):
            jtype = 'fixed'
            jname = f"fix_{parent}_{name}"
            axis  = [0, 0, 1]; rng = None
        else:
            jname = mj_j.get('name', f"j_{parent}_{name}")
            mjt   = mj_j.get('type', 'hinge')
            if mjt in ('hinge', 'revolute'):   jtype = 'revolute'
            elif mjt == 'slide':               jtype = 'prismatic'
            else:                              jtype = 'fixed'
            axis = vec(mj_j.get('axis', '0 0 1'), pad=3)
            rng  = mj_j.get('range')

        jt = ET.Element('joint', {'name': jname, 'type': jtype})
        ET.SubElement(jt, 'parent', {'link': parent})
        ET.SubElement(jt, 'child',  {'link': name})
        ET.SubElement(jt, 'origin',
                      {'xyz': f"{pos_b[0]} {pos_b[1]} {pos_b[2]}",
                       'rpy':  f"{rpy_b[0]} {rpy_b[1]} {rpy_b[2]}"})

        if jtype in ('revolute', 'prismatic'):
            ET.SubElement(jt, 'axis', {'xyz': f"{axis[0]} {axis[1]} {axis[2]}"})
            if rng:
                lo, hi = rng.split()
            else:
                lo, hi = ("-2.967", "2.967") if jtype == 'revolute' else ("-0.5", "0.5")
            ET.SubElement(jt, 'limit',
                          {'lower': lo, 'upper': hi,
                           'effort': '250', 'velocity': '2'})
        joints.append(jt)

    for child in body.findall('body'):
        process_body(child, name)

# ---------------- run --------------------------------------------------------
root_body = root.find('./worldbody/body')
process_body(root_body, None)

robot = ET.Element('robot', {'name': 'panda_from_mjcf'})
robot.extend(links.values());  robot.extend(joints)
indent(robot)
ET.ElementTree(robot).write(out_path, encoding='utf-8', xml_declaration=True)
print("Saved", out_path)
