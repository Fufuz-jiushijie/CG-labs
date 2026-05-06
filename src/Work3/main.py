import taichi as ti

ti.init(arch=ti.gpu)

# 外部固定参数
bg_color = (0.5, 0.5, 0.3)
RES = (1000, 800)
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100
POINT_COLOR = (1.0, 0.0, 0.0)
LINE_COLOR = (0.0, 0.2, 0.1)
# 曲线颜色（内核内用标量，便于与背景 alpha 混合）
CURVE_R = 0.0
CURVE_G = 0.5
CURVE_B = 0.5

MODE_BEZIER = 0
MODE_BSPLINE = 1

count = 0
use_antialiasing = True
curve_mode = MODE_BEZIER

pixels = ti.Vector.field(3, ti.f32, shape=RES)
pixels.fill(bg_color)
control_points = ti.Vector.field(2, ti.f32, shape=MAX_CONTROL_POINTS)
curve_points = ti.Vector.field(2, ti.f32, NUM_SEGMENTS)


def clear_canvas():
    global count
    pixels.fill(bg_color)
    control_points.fill(ti.Vector([-1.0, -1.0]))
    count = 0


@ti.func
def de_Casteljau(t: ti.f32, current_count: int) -> ti.types.vector(2, ti.f32):
    tmp = ti.Matrix.zero(ti.f32, MAX_CONTROL_POINTS, 2)
    for i in range(current_count):
        tmp[i, 0] = control_points[i].x
        tmp[i, 1] = control_points[i].y
    for step in range(1, current_count):
        for i in range(current_count - step):
            tmp[i, 0] = tmp[i, 0] * (1.0 - t) + tmp[i + 1, 0] * t
            tmp[i, 1] = tmp[i, 1] * (1.0 - t) + tmp[i + 1, 1] * t
    return ti.Vector([tmp[0, 0], tmp[0, 1]])


@ti.func
def uniform_cubic_bspline_point(seg_start: int, u: ti.f32) -> ti.types.vector(2, ti.f32):
    """均匀三次 B 样条单段，局部参数 u∈[0,1]，使用标准基矩阵 U·M·[P0..P3]^T。"""
    P0 = control_points[seg_start]
    P1 = control_points[seg_start + 1]
    P2 = control_points[seg_start + 2]
    P3 = control_points[seg_start + 3]
    u2 = u * u
    u3 = u2 * u
    # M = 1/6 * [[-1,3,-3,1],[3,-6,3,0],[-3,0,3,0],[1,4,1,0]]
    ax = (-P0.x + 3.0 * P1.x - 3.0 * P2.x + P3.x) / 6.0
    bx = (3.0 * P0.x - 6.0 * P1.x + 3.0 * P2.x) / 6.0
    cx = (-3.0 * P0.x + 3.0 * P2.x) / 6.0
    dx = (P0.x + 4.0 * P1.x + P2.x) / 6.0
    ay = (-P0.y + 3.0 * P1.y - 3.0 * P2.y + P3.y) / 6.0
    by = (3.0 * P0.y - 6.0 * P1.y + 3.0 * P2.y) / 6.0
    cy = (-3.0 * P0.y + 3.0 * P2.y) / 6.0
    dy = (P0.y + 4.0 * P1.y + P2.y) / 6.0
    x = ax * u3 + bx * u2 + cx * u + dx
    y = ay * u3 + by * u2 + cy * u + dy
    return ti.Vector([x, y])


@ti.func
def splat_subpixel(
    pt: ti.types.vector(2, ti.f32),
    r: ti.f32,
    g: ti.f32,
    b: ti.f32,
    antialias: ti.i32,
):
    """亚像素 splat：关闭 AA 时点亮最近像素；开启时对 3×3 邻域按与像素中心距离做高斯权重混合。"""
    if antialias == 0:
        px = int(pt.x)
        py = int(pt.y)
        if 0 <= px < RES[0] and 0 <= py < RES[1]:
            pixels[px, py] = ti.Vector([r, g, b])
    else:
        cx = pt.x
        cy = pt.y
        ix = int(ti.floor(cx))
        iy = int(ti.floor(cy))
        sigma_sq = 0.81
        col = ti.Vector([r, g, b])
        for di, dj in ti.ndrange((-1, 2), (-1, 2)):
            px = ix + di
            py = iy + dj
            if 0 <= px < RES[0] and 0 <= py < RES[1]:
                pcx = ti.cast(px, ti.f32) + 0.5
                pcy = ti.cast(py, ti.f32) + 0.5
                dx = cx - pcx
                dy = cy - pcy
                dist_sq = dx * dx + dy * dy
                w = ti.exp(-dist_sq / (2.0 * sigma_sq))
                w = ti.min(w, 1.0)
                old = pixels[px, py]
                pixels[px, py] = old * (1.0 - w) + col * w


@ti.func
def draw_bezier_curve(current_count: int, antialias: ti.i32):
    if current_count >= 2:
        for i in range(NUM_SEGMENTS):
            t = i / ti.max(float(NUM_SEGMENTS - 1), 1.0)
            pt = de_Casteljau(t, current_count)
            curve_points[i] = pt
            splat_subpixel(pt, CURVE_R, CURVE_G, CURVE_B, antialias)


@ti.func
def draw_bspline_curve(current_count: int, antialias: ti.i32):
    """n≥4 时共 n-3 段均匀三次 B 样条；采样均匀分配到各段。"""
    if current_count >= 4:
        num_seg = current_count - 3
        for i in range(NUM_SEGMENTS):
            t_global = i / ti.max(float(NUM_SEGMENTS - 1), 1.0)
            s = t_global * ti.cast(num_seg, ti.f32)
            seg = int(ti.floor(s))
            if seg >= num_seg:
                seg = num_seg - 1
            u = s - ti.cast(seg, ti.f32)
            if seg == num_seg - 1 and u > 1.0 - 1e-5:
                u = 1.0
            pt = uniform_cubic_bspline_point(seg, u)
            curve_points[i] = pt
            splat_subpixel(pt, CURVE_R, CURVE_G, CURVE_B, antialias)


@ti.func
def draw_line(idx0: int, idx1: int, antialias: ti.i32):
    p0 = control_points[idx0]
    p1 = control_points[idx1]
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    steps = ti.max(ti.abs(dx), ti.abs(dy))
    if steps > 0.0:
        x_inc = dx / steps
        y_inc = dy / steps
        Lr = LINE_COLOR[0]
        Lg = LINE_COLOR[1]
        Lb = LINE_COLOR[2]
        for i in range(int(steps) + 1):
            curr_x = p0.x + ti.cast(i, ti.f32) * x_inc
            curr_y = p0.y + ti.cast(i, ti.f32) * y_inc
            if 0 <= curr_x < RES[0] and 0 <= curr_y < RES[1]:
                splat_subpixel(ti.Vector([curr_x, curr_y]), Lr, Lg, Lb, antialias)


@ti.kernel
def update_control_point(count: int, mouse_pos: ti.types.vector(2, ti.f32)):
    control_points[count] = mouse_pos


@ti.func
def draw_point(px: int, py: int):
    for i, j in ti.ndrange((-2, 3), (-2, 3)):
        if 0 <= px + i < RES[0] and 0 <= py + j < RES[1]:
            pixels[px + i, py + j] = ti.Vector(
                [POINT_COLOR[0], POINT_COLOR[1], POINT_COLOR[2]]
            )


@ti.kernel
def render_scene(n: int, mode: ti.i32, antialias: ti.i32):
    if mode == 0:
        if n >= 2:
            draw_bezier_curve(n, antialias)
    else:
        if n >= 4:
            draw_bspline_curve(n, antialias)
    for i in range(n):
        if i >= 1:
            draw_line(i - 1, i, antialias)
        px = int(control_points[i].x)
        py = int(control_points[i].y)
        draw_point(px, py)


def redraw():
    pixels.fill(bg_color)
    aa = 1 if use_antialiasing else 0
    mode = MODE_BSPLINE if curve_mode == MODE_BSPLINE else MODE_BEZIER
    render_scene(count, mode, aa)


def main():
    global count, use_antialiasing, curve_mode
    title = "LMB 加控制点 | c 清空 | a 反走样开关 | b 切换曲线模式"
    window = ti.ui.Window(title, RES, fps_limit=60, pos=(100, 100))
    while window.running:
        while window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SHIFT:
                print("Shift：配合说明见窗口标题。")

            elif window.event.key == ti.ui.LMB:
                if count < MAX_CONTROL_POINTS:
                    mouse_pos_norm = window.get_cursor_pos()
                    mouse_pos = (
                        mouse_pos_norm[0] * RES[0],
                        mouse_pos_norm[1] * RES[1],
                    )
                    update_control_point(count, mouse_pos)
                    count += 1
                    print(f"控制点数: {count}")
                    redraw()
                else:
                    print(f"已达上限 {MAX_CONTROL_POINTS} 个点，按 c 清空")

            elif window.event.key == "c":
                print("画布已清空。")
                clear_canvas()

            elif window.event.key == "a":
                use_antialiasing = not use_antialiasing
                print(f"反走样: {'开' if use_antialiasing else '关'}")
                if count > 0:
                    redraw()

            elif window.event.key == "b":
                if curve_mode == MODE_BEZIER:
                    curve_mode = MODE_BSPLINE
                    print("曲线模式: 均匀三次 B 样条（至少 4 个控制点可见曲线）")
                else:
                    curve_mode = MODE_BEZIER
                    print("曲线模式: Bézier（至少 2 个控制点可见曲线）")
                if count > 0:
                    redraw()

            else:
                print(f"按键: {window.event.key}（见标题栏说明）")

        canvas = window.get_canvas()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()
