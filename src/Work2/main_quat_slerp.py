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


def write_rotated_vertices(vertices: ti.Field, base_vertices: Sequence[Vec3], q: Quat):
    for i in range(8):
        rx, ry, rz = quat_rotate_vec3(q, base_vertices[i])
        vertices[i] = (rx, ry, rz)


def cube_screen_points(vertices, screen_coords, state, base_vertices, q, translation):
    """旋转 + 平移并做 MVP 变换，返回该立方体 8 个顶点的屏幕坐标 [(x, y, z_ndc), ...]。"""
    write_rotated_vertices(vertices, base_vertices, q)
    state.translation = ti.math.vec3([translation[0], translation[1], translation[2]])
    compute_transform(
        vertices, screen_coords,
        state.eye_pos, state.target_pos, state.up,
        state.translation, state.rotation_angles, state.scale,
        state.fov_y, state.aspect_ratio, state.z_near, state.z_far,
    )
    return [
        (float(screen_coords[i][0]), float(screen_coords[i][1]), float(screen_coords[i][2]))
        for i in range(8)
    ]


def render_scene(gui, cube_list, faces, edges):
    """全局画家算法：把所有立方体的面/边按深度统一排序后绘制，
    使共面的立方体按真实深度互相遮挡，而非按绘制先后顺序覆盖。"""
    face_batch = []  # (z_avg, p0, p1, p2, p3, fill_color)
    edge_batch = []  # (z_avg, pa, pb, color, radius)

    for pts, face_color, alpha, edge_color, edge_radius in cube_list:
        fill = blend_over_white(face_color, alpha)
        for (a, b, c, d) in faces:
            z = (pts[a][2] + pts[b][2] + pts[c][2] + pts[d][2]) / 4.0
            face_batch.append((z, pts[a], pts[b], pts[c], pts[d], fill))
        for (i, j) in edges:
            z = (pts[i][2] + pts[j][2]) / 2.0
            edge_batch.append((z, pts[i], pts[j], edge_color, edge_radius))

    # 从远到近绘制
    face_batch.sort(key=lambda f: f[0], reverse=True)
    for _, p0, p1, p2, p3, fill in face_batch:
        v0 = ti.Vector([p0[0], p0[1]])
        v1 = ti.Vector([p1[0], p1[1]])
        v2 = ti.Vector([p2[0], p2[1]])
        v3 = ti.Vector([p3[0], p3[1]])
        gui.triangle(v0, v1, v2, color=fill)
        gui.triangle(v0, v2, v3, color=fill)

    edge_batch.sort(key=lambda e: e[0], reverse=True)
    for _, pa, pb, color, radius in edge_batch:
        gui.line(ti.Vector([pa[0], pa[1]]), ti.Vector([pb[0], pb[1]]), radius=radius, color=color)


def main():
    ti.init(arch=ti.gpu)

    state = RenderState(shape_type="cube")
    
    # 正对 xOy 旋转平面：相机放在 +Z 轴上，沿 -Z 方向直视，不俯仰/不倾斜
    state.eye_pos = ti.math.vec3([0.0, 0.0, 9.0])

    # target_pos: 看向原点（轨迹中心），视线垂直于 xOy 平面
    state.target_pos = ti.math.vec3([0.0, 0.0, 0.0])

    # up: 世界坐标系 Y 轴朝上（屏幕上 +y 向上，+x 向右）
    state.up = ti.math.vec3([0.0, 1.0, 0.0])
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

    # 端点姿态：绕 Z 轴（垂直于 xOy 旋转平面）旋转，使立方体的同一面始终朝向圆心。
    # 位置极角 ang = pi*(1-t)，让朝向角等于 ang，则 slerp(q0,q1,t) 恰好 = 绕 Z 转 ang，
    # 立方体随半径转动（-X 面始终指向圆心），而不是原来的绕 Y 自转。
    q0: Quat = quat_from_axis_angle((0.0, 0.0, 1.0), math.pi)  # 左端点 t=0，ang=pi
    q1: Quat = (1.0, 0.0, 0.0, 0.0)                            # 右端点 t=1，ang=0（单位四元数）

    # 颜色（淡蓝端点 / 淡黄路径 / 橙色匀速移动体）
    blue = 0x7EC8FF
    yellow = 0xFFF2A1
    orange = 0xFF8C42

    gui = ti.GUI("Quaternion SLERP Cube (Work2)", res=(800, 600))

    t_time = 0.0
    dt = 1.0 / 60.0
    duration = 2.5  # 秒

    # 旋转完全由四元数烘焙进顶点，模型矩阵的欧拉角保持为 0
    state.rotation_angles = ti.math.vec3([0.0, 0.0, 0.0])

    # 端点立方体放在 x 负/正半轴；插值体沿上半圆弧摆放
    endpoint_x = 1.5
    R = endpoint_x

    def arc_pos(t: float) -> Tuple[float, float]:
        ang = math.pi * (1.0 - t)  # t=0 -> pi (左)，t=1 -> 0 (右)
        return R * math.cos(ang), R * math.sin(ang) * 1

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
        # ping-pong: 0->1->0，t_anim 关于时间分段线性（匀速）
        phase = (t_time / duration) % 2.0
        t_anim = phase if phase <= 1.0 else (2.0 - phase)

        # 收集本帧所有立方体（端点 + 静态路径 + 匀速移动体），稍后统一按深度排序绘制。
        # 这样所有立方体共用同一套深度遮挡关系，黄/橙不会因为“后画”就盖在蓝色上面。
        # cube_list 元素：(屏幕坐标点列表, 面颜色, alpha, 边颜色, 边半径)
        cube_list = []

        # 左端点（t=0，淡蓝半透明）
        pts = cube_screen_points(vertices, screen_coords, state, base_vertices, q0, (-endpoint_x, 0.0, 0.0))
        cube_list.append((pts, blue, 0.35, 0x2B2B2B, 1))

        # 右端点（t=1，淡蓝半透明）
        pts = cube_screen_points(vertices, screen_coords, state, base_vertices, q1, (endpoint_x, 0.0, 0.0))
        cube_list.append((pts, blue, 0.35, 0x2B2B2B, 1))

        # 5 个静态插值立方体（淡黄半透明）：始终显示完整插值路径，作为参考轨迹
        for k in range(1, 6):
            t_k = k / 6.0
            qk = quat_slerp(q0, q1, t_k)
            x, y = arc_pos(t_k)
            pts = cube_screen_points(vertices, screen_coords, state, base_vertices, qk, (x, y, 0.0))
            cube_list.append((pts, yellow, 0.18, 0x9A9A9A, 1))

        # 单个匀速移动的立方体：t = t_anim 直接驱动 slerp（恒定角速度，沿弧匀速）
        q_move = quat_slerp(q0, q1, t_anim)
        x, y = arc_pos(t_anim)
        pts = cube_screen_points(vertices, screen_coords, state, base_vertices, q_move, (x, y, 0.0))
        cube_list.append((pts, orange, 0.85, 0x2B2B2B, 2))

        render_scene(gui, cube_list, faces, state.edges)

        gui.show()


if __name__ == "__main__":
    main()

