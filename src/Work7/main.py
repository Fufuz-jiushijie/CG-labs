import taichi as ti

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# 物理与网格参数
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长 (半隐式稳定上限约 0.01，当前远低于此)
k_s = 10000.0      # 结构弹簧劲度系数
k_s_shear = 8000.0 # 剪切弹簧劲度系数 (抗错切，可略软于结构弹簧)
k_s_bend = 2000.0  # 弯曲弹簧劲度系数 (抗折叠，通常更软)
k_d = 5.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限，防止数值爆炸

# 选做：球体碰撞参数
# 布料从 j=0 边 (z=-0.5) 的两角垂下，悬挂平面大致在 z≈-0.5；
# 把球心放在该平面上、竖直方向中段，保证布料能真正搭到球上。
sphere_center = ti.Vector([0.0, 0.3, -0.5])  # 球心
sphere_radius = 0.25                          # 球半径
collision_margin = 0.01                       # 碰撞留白，避免布料穿插/闪烁

# 定义 Taichi 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)  # 隐式欧拉专用力场

# 弹簧数据场 (结构 + 剪切 + 弯曲，预留 8*N*N 容量)
max_springs = N * N * 8
spring_indices = ti.field(dtype=int, shape=max_springs * 2) # 用于渲染画线
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_ks = ti.field(dtype=float, shape=max_springs)        # 每根弹簧的劲度系数
num_springs = ti.field(dtype=int, shape=())

# 选做：球体渲染位置 (单独 field 供 GGUI 绘制)
sphere_pos = ti.Vector.field(3, dtype=float, shape=1)

# ============ 初始化 (拆分为多个 kernel 保证 GPU 同步) ============

@ti.kernel
def init_positions():
    """初始化质点位置与固定状态"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        # 将布料放置在三维空间的 XY 平面上
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定第一排的两个角点
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.func
def add_spring(a: int, b: int, ks: float):
    """向弹簧数组追加一根弹簧，自然长度取当前两点间距"""
    c = ti.atomic_add(num_springs[None], 1)
    spring_pairs[c] = ti.Vector([a, b])
    spring_lengths[c] = (x[a] - x[b]).norm()
    spring_ks[c] = ks

@ti.kernel
def init_springs(enable_shear: int, enable_bending: int):
    """初始化三类弹簧：
       - 结构 (Structural)：水平/垂直相邻，撑起基本网格形状
       - 剪切 (Shear)：对角相邻，抵抗布料错切变形
       - 弯曲 (Bending)：跨一个点的同行/同列，抵抗布料过度折叠
    """
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        # ---- 结构弹簧 ----
        if i < N - 1:
            add_spring(idx, (i + 1) * N + j, k_s)
        if j < N - 1:
            add_spring(idx, i * N + (j + 1), k_s)
        # ---- 剪切弹簧 (两条对角线) ----
        if enable_shear == 1:
            if i < N - 1 and j < N - 1:
                add_spring(idx, (i + 1) * N + (j + 1), k_s_shear)
            if i < N - 1 and j > 0:
                add_spring(idx, (i + 1) * N + (j - 1), k_s_shear)
        # ---- 弯曲弹簧 (隔一个点相连) ----
        if enable_bending == 1:
            if i < N - 2:
                add_spring(idx, (i + 2) * N + j, k_s_bend)
            if j < N - 2:
                add_spring(idx, i * N + (j + 2), k_s_bend)

@ti.kernel
def init_spring_indices():
    """同步渲染索引"""
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

@ti.kernel
def init_sphere():
    """把碰撞球心写入渲染 field"""
    sphere_pos[0] = sphere_center

def init_cloth(enable_shear=True, enable_bending=True):
    """从 Python 层按顺序调用各初始化 kernel，确保 GPU 同步"""
    num_springs[None] = 0  # 重置弹簧计数，防止重复调用时累加
    init_positions()
    init_springs(1 if enable_shear else 0, 1 if enable_bending else 0)
    init_spring_indices()
    init_sphere()

# ============ 合并的力计算函数 (ti.func 内联到 kernel 中，减少启动开销) ============

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """计算所有力 (重力 + 阻尼 + 弹簧力)，使用 ti.func 内联到调用它的 kernel 中"""
    # 第一阶段：清空受力，施加重力与阻尼 (每个质点独立，无冲突)
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    # Taichi 保证同一 kernel 内的多个顶层 for 循环顺序执行 (GPU 同步屏障)
    # 第二阶段：累加弹簧力 (使用 atomic_add 保证多线程安全)
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -spring_ks[i] * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    """速度钳制，防止数值爆炸"""
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def handle_sphere_collision(pos: ti.template(), vel: ti.template(), enabled: int):
    """球体碰撞处理：质点进入球内时推回球面，并消除朝球心的法向速度。
       enabled 作为整型开关传入，保证顶层 for 循环仍可并行。"""
    surface = sphere_radius + collision_margin
    for i in range(N * N):
        if enabled == 1 and is_fixed[i] == 0:
            diff = pos[i] - sphere_center
            dist = diff.norm()
            if dist < surface:
                n = diff / (dist + 1e-9)
                pos[i] = sphere_center + n * surface  # 投影回球面外侧
                vn = vel[i].dot(n)
                if vn < 0.0:
                    vel[i] -= vn * n  # 去掉指向球心的速度分量

# ============ 合并的积分 kernel (每步仅 1 次 kernel 启动) ============

@ti.kernel
def step_explicit(collide: int):
    """显式欧拉 (Explicit Euler) - 极易发散
       全部计算合并在单个 kernel 中，最小化 GPU 启动开销"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt          # 用旧速度更新位置
            v[i] += (f[i] / mass) * dt  # 用旧力更新速度
            clamp_velocity(v, i)
    handle_sphere_collision(x, v, collide)

@ti.kernel
def step_semi_implicit(collide: int):
    """半隐式欧拉 (Semi-Implicit Euler) - 相对稳定
       全部计算合并在单个 kernel 中，最小化 GPU 启动开销"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt  # 先更新速度
            clamp_velocity(v, i)
            x[i] += v[i] * dt           # 用新速度更新位置
    handle_sphere_collision(x, v, collide)

@ti.kernel
def step_implicit_iter(collide: int):
    """隐式欧拉 (Implicit Euler) - 使用定点迭代法近似求解
       全部计算合并在单个 kernel 中 (ti.static 展开迭代)"""
    # 1. 复制当前状态到预测场
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    # 2. 定点迭代求解未来状态 (ti.static 在编译期展开，无循环开销)
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    # 3. 将收敛后的状态写回
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]
    handle_sphere_collision(x, v, collide)

# ============ 主函数 ============

def main():
    # 建立 GGUI 窗口
    window = ti.ui.Window("Games101 - Mass Spring System", (800,800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1 # 0: 显式, 1: 半隐式, 2: 隐式
    paused = False
    enable_shear = True       # 选做：剪切弹簧
    enable_bending = True     # 选做：弯曲弹簧
    enable_collision = True   # 选做：球体碰撞

    # 初次构建（含选做弹簧）
    init_cloth(enable_shear, enable_bending)

    while window.running:
        # =========== GUI 控制面板 ===========
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.42, 0.56)

        window.GUI.text("Integration Method:")

        # 方法选择按钮 - 当前选中的方法会有标记
        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth(enable_shear, enable_bending)  # 切换方法时重置，防止从不稳定状态继续
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth(enable_shear, enable_bending)
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth(enable_shear, enable_bending)

        window.GUI.text("")  # 空行分隔
        window.GUI.text("Springs (rebuild on toggle):")

        # 剪切 / 弯曲弹簧开关（切换后需重建弹簧）
        shear_label = ("[x] " if enable_shear else "[ ] ") + "Shear Springs"
        if window.GUI.button(shear_label):
            enable_shear = not enable_shear
            init_cloth(enable_shear, enable_bending)
        bend_label = ("[x] " if enable_bending else "[ ] ") + "Bending Springs"
        if window.GUI.button(bend_label):
            enable_bending = not enable_bending
            init_cloth(enable_shear, enable_bending)

        window.GUI.text("")  # 空行分隔

        # 球体碰撞开关（无需重建）
        collide_label = ("[x] " if enable_collision else "[ ] ") + "Sphere Collision"
        if window.GUI.button(collide_label):
            enable_collision = not enable_collision

        window.GUI.text("")  # 空行分隔

        # 暂停/恢复按钮
        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused

        # 重置按钮
        if window.GUI.button("Reset Cloth"):
            init_cloth(enable_shear, enable_bending)

        window.GUI.end()
        # ====================================

        collide_flag = 1 if enable_collision else 0
        if not paused:
            # 每帧物理子步更新 (40步 × 5e-4 = 0.02s/帧 ≈ 实时速度)
            for _ in range(40):
                if current_method == 0:
                    step_explicit(collide_flag)
                elif current_method == 1:
                    step_semi_implicit(collide_flag)
                elif current_method == 2:
                    step_implicit_iter(collide_flag)

        # 渲染场景
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        # 绘制网格顶点和弹簧线框
        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        # 绘制碰撞球
        if enable_collision:
            scene.particles(sphere_pos, radius=sphere_radius, color=(0.9, 0.45, 0.3))

        canvas.scene(scene)

        window.show()

if __name__ == '__main__':
    main()