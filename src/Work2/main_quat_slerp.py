import math
from typing import Tuple, List, Sequence

import taichi as ti

from src.Work2.class_renderstate import RenderState
from src.Work2.mvp import compute_transform

Quat = Tuple[float, float, float, float]  # (w, x, y, z)
Vec3 = Tuple[float, float, float]


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def quat_normalize(q: Quat) -> Quat:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    inv = 1.0 / n
    return (w * inv, x * inv, y * inv, z * inv)


def quat_from_axis_angle(axis: Vec3, angle_rad: float) -> Quat:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    inv = 1.0 / n
    ax, ay, az = ax * inv, ay * inv, az * inv
    half = 0.5 * angle_rad
    s = math.sin(half)
    # [w, x, y, z]
    return quat_normalize((math.cos(half), ax * s, ay * s, az * s))


def quat_slerp(q0: Quat, q1: Quat, t: float) -> Quat:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    t = _clamp(t, 0.0, 1.0)
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]

    # 取最短弧
    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot

    dot = _clamp(dot, -1.0, 1.0)

    # 很接近时退化为 lerp，避免数值不稳定
    if dot > 0.9995:
        w = q0[0] + t * (q1[0] - q0[0])
        x = q0[1] + t * (q1[1] - q0[1])
        y = q0[2] + t * (q1[2] - q0[2])
        z = q0[3] + t * (q1[3] - q0[3])
        return quat_normalize((w, x, y, z))

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (
        s0 * q0[0] + s1 * q1[0],
        s0 * q0[1] + s1 * q1[1],
        s0 * q0[2] + s1 * q1[2],
        s0 * q0[3] + s1 * q1[3],
    )


def quat_rotate_vec3(q: Quat, v: Vec3) -> Vec3:
    # v' = v + 2 * cross(qv, cross(qv, v) + w*v)
    w, qx, qy, qz = quat_normalize(q)
    vx, vy, vz = v

    # t = 2 * cross(qv, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    # v' = v + w*t + cross(qv, t)
    cx = qy * tz - qz * ty
    cy = qz * tx - qx * tz
    cz = qx * ty - qy * tx

    return (vx + w * tx + cx, vy + w * ty + cy, vz + w * tz + cz)


def hex_to_rgb(color: int) -> Tuple[int, int, int]:
    return (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF


def rgb_to_hex(r: int, g: int, b: int) -> int:
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return (r << 16) + (g << 8) + b


def blend_over_white(face_color_hex: int, alpha: float) -> int:
    # GUI 没有 alpha；用和白背景混合模拟透明
    fr, fg, fb = hex_to_rgb(face_color_hex)
    br, bg, bb = 255, 255, 255
    r = alpha * fr + (1.0 - alpha) * br
    g = alpha * fg + (1.0 - alpha) * bg
    b = alpha * fb + (1.0 - alpha) * bb
    return rgb_to_hex(r, g, b)


def draw_cube_faces(
    gui: ti.GUI,
    screen_coords: ti.Field,
    faces: List[Tuple[int, int, int, int]],
    color_hex: int,
    alpha: float,
):
    # 按 NDC z 平均值从远到近画（更适合半透明）
    face_with_depth = []
    for (a, b, c, d) in faces:
        z_avg = float((screen_coords[a][2] + screen_coords[b][2] + screen_coords[c][2] + screen_coords[d][2]) / 4.0)
        face_with_depth.append((z_avg, (a, b, c, d)))
    face_with_depth.sort(key=lambda x: x[0], reverse=True)

    fill_color = blend_over_white(color_hex, alpha)

    for _, (a, b, c, d) in face_with_depth:
        p0 = ti.Vector([screen_coords[a][0], screen_coords[a][1]])
        p1 = ti.Vector([screen_coords[b][0], screen_coords[b][1]])
        p2 = ti.Vector([screen_coords[c][0], screen_coords[c][1]])
        p3 = ti.Vector([screen_coords[d][0], screen_coords[d][1]])
        gui.triangle(p0, p1, p2, color=fill_color)
        gui.triangle(p0, p2, p3, color=fill_color)


def draw_cube_edges(
    gui: ti.GUI,
    screen_coords: ti.Field,
    edges: Sequence[Tuple[int, int]],
    color: int = 0x000000,
    radius: int = 1,
):
    for i, j in edges:
        p1 = ti.Vector([screen_coords[i][0], screen_coords[i][1]])
        p2 = ti.Vector([screen_coords[j][0], screen_coords[j][1]])
        gui.line(p1, p2, radius=radius, color=color)


def write_rotated_vertices(vertices: ti.Field, base_vertices: Sequence[Vec3], q: Quat):
    for i in range(8):
        rx, ry, rz = quat_rotate_vec3(q, base_vertices[i])
        vertices[i] = (rx, ry, rz)


def main():
    ti.init(arch=ti.cpu)

    state = RenderState(shape_type="cube")
    
    state.eye_pos = ti.math.vec3([0.0, 3.0, 9.0]) 
    
    # target_pos: 观察目标 (y=0.4 是为了稍微抬高视点，因为圆弧是在y=0到y≈0.8之间)
    state.target_pos = ti.math.vec3([0.0, 0.0, 1.0]) 
    
    # up: 相机的向上方向 (保持世界坐标系的Y轴向上)
    state.up = ti.math.vec3([0.0, -1.0, 0.0])
    # 顶点 field（每次绘制一个立方体就覆盖写入一次）
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=state.num_vertices)
    # screen_coords: [x_screen, y_screen, z_ndc]
    screen_coords = ti.Vector.field(3, dtype=ti.f32, shape=state.num_vertices)

    # 立方体局部顶点（与 Work2 的编号保持一致）
    s = 0.5
    base_vertices: List[Vec3] = [
        (s, s, s),
        (-s, s, s),
        (-s, -s, s),
        (s, -s, s),
        (s, s, -s),
        (-s, s, -s),
        (-s, -s, -s),
        (s, -s, -s),
    ]

    # 6 个面（每面 4 个点）
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 4, 5, 1),
        (3, 7, 6, 2),
        (1, 5, 6, 2),
        (0, 4, 7, 3),
    ]

    # 端点姿态：从 +Z 方向看，绕 Y 轴旋转（示例：0 -> 180°）
    q0: Quat = (1.0, 0.0, 0.0, 0.0)
    q1: Quat = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi)

    # 颜色（淡蓝 / 淡黄）
    blue = 0x7EC8FF
    yellow = 0xFFF2A1

    gui = ti.GUI("Quaternion SLERP Cube (Work2)", res=(800, 600))

    t_time = 0.0
    dt = 1.0 / 60.0
    duration = 2.5  # 秒

    while gui.running:
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            # elif gui.event.key == 'a':
            #     state.rotation_angles[2] += 0.2
            # elif gui.event.key == 'd':
            #     state.rotation_angles[2] -= 0.2
            # # 绕x轴旋转
            # elif gui.event.key == 'w':
            #     state.rotation_angles[0] += 0.2
            # elif gui.event.key == 's':
            #     state.rotation_angles[0] -= 0.2
            # elif gui.event.key == 'q':
            #     state.rotation_angles[1] += 0.2
            # elif gui.event.key == 'e':
            #     state.rotation_angles[1] -= 0.2
                
            # # 相机位置移动
            # elif gui.event.key == 'i':
            #     state.eye_pos[2] -= 0.2
            # elif gui.event.key == 'k':
            #     state.eye_pos[2] += 0.2
            # elif gui.event.key == 'j':
            #     state.eye_pos[0] -= 0.2
            # elif gui.event.key == 'l':
            #     state.eye_pos[0] += 0.2
            # elif gui.event.key == 'u':
            #     state.eye_pos[1] += 0.2
            # elif gui.event.key == 'o':
            #     state.eye_pos[1] -= 0.2
        # 白色背景
        # gui.clear(0xFFFFFF)

        t_time += dt
        # ping-pong: 0->1->0
        phase = (t_time / duration) % 2.0
        t_anim = phase if phase <= 1.0 else (2.0 - phase)

        # 端点立方体：放在 x 负/正半轴
        endpoint_x = 1.5

        # 画左端点（t=0，淡蓝半透明）
        q_left = q0
        write_rotated_vertices(vertices, base_vertices, q_left)
        state.translation = ti.math.vec3([-endpoint_x, 0.0, 0.0])
        state.rotation_angles = ti.math.vec3([0.0, 0.0, 0.0])
        compute_transform(
            vertices,
            screen_coords,
            state.eye_pos, state.target_pos, state.up,
            state.translation, state.rotation_angles, state.scale,
            state.fov_y, state.aspect_ratio, state.z_near, state.z_far
        )
        draw_cube_faces(gui, screen_coords, faces, blue, alpha=0.35)
        draw_cube_edges(gui, screen_coords, state.edges, color=0x2B2B2B, radius=1)

        # 画右端点（t=1，淡蓝半透明）
        q_right = q1
        write_rotated_vertices(vertices, base_vertices, q_right)
        state.translation = ti.math.vec3([endpoint_x, 0.0, 0.0])
        compute_transform(
            vertices,
            screen_coords,
            state.eye_pos, state.target_pos, state.up,
            state.translation, state.rotation_angles, state.scale,
            state.fov_y, state.aspect_ratio, state.z_near, state.z_far
        )
        draw_cube_faces(gui, screen_coords, faces, blue, alpha=0.35)
        draw_cube_edges(gui, screen_coords, state.edges, color=0x2B2B2B, radius=1)

        # 5 个插值立方体（淡黄半透明），沿上半圆弧摆放，t 随动画进度展开
        for k in range(1, 6):
            t_k = (k / 6.0) * t_anim
            qk = quat_slerp(q0, q1, t_k)
            write_rotated_vertices(vertices, base_vertices, qk)

            # 上半圆弧：从 (-R,0) -> (R,0)，中间 y>0
            R = endpoint_x
            ang = math.pi * (1.0 - t_k)  # t=0 -> pi (左)，t=1 -> 0 (右)
            x = R * math.cos(ang)
            y = R * math.sin(ang) * 0.55
            state.translation = ti.math.vec3([x, y, 0.0])
            compute_transform(
                vertices,
                screen_coords,
                state.eye_pos, state.target_pos, state.up,
                state.translation, state.rotation_angles, state.scale,
                state.fov_y, state.aspect_ratio, state.z_near, state.z_far
            )
            draw_cube_faces(gui, screen_coords, faces, yellow, alpha=0.30)
            draw_cube_edges(gui, screen_coords, state.edges, color=0x2B2B2B, radius=1)

        gui.show()


if __name__ == "__main__":
    main()

