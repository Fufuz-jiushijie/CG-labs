import taichi as ti
from src.Work2.class_renderstate import RenderState
from src.Work2.mvp import compute_transform

def main():
    ti.init(arch=ti.gpu)

    state = RenderState(shape_type="cube")
    
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=state.num_vertices)
    # screen_coords: [x_screen, y_screen, z_ndc]
    screen_coords = ti.Vector.field(3, dtype=ti.f32, shape=state.num_vertices)
    
    if state.shape_type == "triangle":
        vertices[0] = [2.0, 0.0, -2.0]
        vertices[1] = [0.0, 2.0, -2.0]
        vertices[2] = [-2.0, 0.0, -2.0]
    elif state.shape_type == "cube":
        s = 0.5
        vertices[0] = [s, s, s]
        vertices[1] = [-s, s, s]
        vertices[2] = [-s, -s, s]
        vertices[3] = [s, -s, s]
        vertices[4] = [s, s, -s]
        vertices[5] = [-s, s, -s]
        vertices[6] = [-s, -s, -s]
        vertices[7] = [s, -s, -s]

    # 渲染模式：1 线框；2 面填充；3 面填充+简单质感
    render_mode = 1
    # 立方体 6 个面（每个面是 4 个顶点索引）
    faces = [
        (0, 1, 2, 3),  # z = +s
        (4, 5, 6, 7),  # z = -s
        (0, 4, 5, 1),  # y = +s
        (3, 7, 6, 2),  # y = -s
        (1, 5, 6, 2),  # x = -s
        (0, 4, 7, 3),  # x = +s
    ]
    # 每个面的基础颜色（不同面不同色）
    face_colors = [
        0xFF5A5A,
        0x5AFF5A,
        0x5A5AFF,
        0xFFF05A,
        0x5AFFF0,
        0xF05AFF,
    ]
    
    gui = ti.GUI("3D Transformation (Taichi)", res=(700, 700))
    
    while gui.running:
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            # 渲染模式切换：1 线框；2 面填充；3 面填充+质感
            elif gui.event.key == '1':
                render_mode = 1
            elif gui.event.key == '2':
                render_mode = 2
            elif gui.event.key == '3':
                render_mode = 3
            # 绕z轴旋转
            elif gui.event.key == 'a':
                state.rotation_angles[2] += 0.2
            elif gui.event.key == 'd':
                state.rotation_angles[2] -= 0.2
            # 绕x轴旋转
            elif gui.event.key == 'w':
                state.rotation_angles[0] += 0.2
            elif gui.event.key == 's':
                state.rotation_angles[0] -= 0.2
            elif gui.event.key == 'q':
                state.rotation_angles[1] += 0.2
            elif gui.event.key == 'e':
                state.rotation_angles[1] -= 0.2
                
            # 相机位置移动
            elif gui.event.key == 'i':
                state.eye_pos[2] -= 0.2
            elif gui.event.key == 'k':
                state.eye_pos[2] += 0.2
            elif gui.event.key == 'j':
                state.eye_pos[0] -= 0.2
            elif gui.event.key == 'l':
                state.eye_pos[0] += 0.2
            elif gui.event.key == 'u':
                state.eye_pos[1] += 0.2
            elif gui.event.key == 'o':
                state.eye_pos[1] -= 0.2
                
            # 视角扩大缩小
            elif gui.event.key == 'm':
                state.fov_y = max(10.0 * ti.math.pi / 180.0, state.fov_y - 0.2)
            elif gui.event.key == 'n':
                state.fov_y = min(120.0 * ti.math.pi / 180.0, state.fov_y + 0.2)
        
        compute_transform(
            vertices,
            screen_coords,
            state.eye_pos, state.target_pos, state.up,
            state.translation, state.rotation_angles, state.scale,
            state.fov_y, state.aspect_ratio, state.z_near, state.z_far
        )

        if state.shape_type == "cube" and render_mode in (2, 3):
            # 面按深度从远到近排序（透明填充更自然）
            face_with_depth = []
            for fi, (a, b, c, d) in enumerate(faces):
                z_avg = (
                    screen_coords[a][2] + screen_coords[b][2] +
                    screen_coords[c][2] + screen_coords[d][2]
                ) / 4.0
                face_with_depth.append((z_avg, fi, (a, b, c, d)))
            face_with_depth.sort(key=lambda x: x[0], reverse=True)

            # 简单“透明度”：通过与黑色背景混合模拟
            bg_r, bg_g, bg_b = 1, 1, 1

            for z_avg, fi, (a, b, c, d) in face_with_depth:
                p0 = ti.Vector([screen_coords[a][0], screen_coords[a][1]])
                p1 = ti.Vector([screen_coords[b][0], screen_coords[b][1]])
                p2 = ti.Vector([screen_coords[c][0], screen_coords[c][1]])
                p3 = ti.Vector([screen_coords[d][0], screen_coords[d][1]])

                base = face_colors[fi]

                if render_mode == 2:
                    alpha = 0.90
                    shade = 1.0
                else:
                    alpha = 0.80
                    # 用深度做一个很轻量的明暗变化（增加体积感）
                    shade = 0.65 - 0.35 * float(z_avg)
                    shade = max(0.30, min(1.00, shade))

                r = int(((base >> 16) & 0xFF) * shade)
                g = int(((base >> 8) & 0xFF) * shade)
                b = int((base & 0xFF) * shade)

                r = int(alpha * r + (1.0 - alpha) * bg_r)
                g = int(alpha * g + (1.0 - alpha) * bg_g)
                b = int(alpha * b + (1.0 - alpha) * bg_b)
                color = (r << 16) + (g << 8) + b

                # 四边形拆成两个三角形
                gui.triangle(p0, p1, p2, color=color)
                gui.triangle(p0, p2, p3, color=color)

            # 叠加线框：边缘更清晰，看起来更“有质感”
            for i, j in state.edges:
                p1 = ti.Vector([screen_coords[i][0], screen_coords[i][1]])
                p2 = ti.Vector([screen_coords[j][0], screen_coords[j][1]])
                gui.line(p1, p2, radius=1, color=0x000000)
        else:
            # 线框模式（或非 cube）
            for i, j in state.edges:
                p1 = ti.Vector([screen_coords[i][0], screen_coords[i][1]])
                p2 = ti.Vector([screen_coords[j][0], screen_coords[j][1]])
                gui.line(p1, p2, radius=1, color=0xFFFFFF)
        
        gui.show()

if __name__ == '__main__':
    main()
