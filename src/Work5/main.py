import taichi as ti

# 初始化 Taichi GPU 后端 (Mac 自动调用 Metal，Win 调用 CUDA/Vulkan)
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
num_samples = ti.field(ti.i32, shape=())  # 每像素采样数（MSAA）
ior = ti.field(ti.f32, shape=())          # 玻璃折射率

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    """
    斯涅尔定律折射。I 为入射方向（指向表面，单位向量），N 为与 I 相对的表面法线，
    eta = n1/n2。返回 (折射方向, 是否发生全反射)。
    """
    refr_dir = ti.Vector([0.0, 0.0, 0.0])
    tir = True  # 默认全反射
    cos_i = -I.dot(N)
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)  # 透射角余弦的平方
    if k >= 0.0:
        tir = False
        refr_dir = normalize(eta * I + (eta * cos_i - ti.sqrt(k)) * N)
    return refr_dir, tir

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0]) # 法线永远朝上
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回: (t, 法线 N, 颜色 color, 材质 mat_id)
    """
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测玻璃球（原红球改为玻璃材质）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.97, 1.0])  # 玻璃的轻微冷色调
        hit_mat = MAT_GLASS

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9]) # 镜面反射基础色
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        # 生成棋盘格纹理
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        # 判断坐标的奇偶性来交替颜色
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3]) # 灰色格子
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8]) # 白色格子

    return min_t, hit_n, hit_c, hit_mat

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        spp = num_samples[None]
        pixel_color = ti.Vector([0.0, 0.0, 0.0])

        # --- MSAA：每个像素内多次随机采样后取平均 ---
        for s in range(spp):
            # 在像素方格内做亚像素抖动（[0,1) 随机偏移）
            jx = ti.random(ti.f32)
            jy = ti.random(ti.f32)
            u = (i + jx - res_x / 2.0) / res_y * 2.0
            v = (j + jy - res_y / 2.0) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])          # 摄像机稍微抬高一点
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))  # 视角微微向下看

            final_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])  # 光线能量吞吐量

            # 迭代式光线追踪（代替递归）
            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)

                # 如果没击中任何物体，加上背景色并结束追踪
                if t > 1e9:
                    final_color += throughput * bg_color
                    break

                p = ro + rd * t

                # 分支 1：镜面反射材质
                if mat_id == MAT_MIRROR:
                    # 生成反射射线，注意必须要加上极其微小的法线偏移（1e-4）防止自相交！
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    # 镜面吸收一部分能量 (反射率 0.8)
                    throughput *= 0.8 * obj_color
                    # 不跳出循环，继续追踪反射射线

                # 分支 2：玻璃（折射 + 全反射）
                elif mat_id == MAT_GLASS:
                    # 根据射线与法线的方向判断当前是“进入”还是“离开”玻璃
                    outward_n = N
                    eta = 1.0 / ior[None]      # 空气 → 玻璃
                    if rd.dot(N) > 0.0:        # 射线在玻璃内部，正在离开
                        outward_n = -N
                        eta = ior[None]        # 玻璃 → 空气

                    refr_dir, tir = refract(rd, outward_n, eta)

                    # 用 Schlick 近似估计菲涅尔反射比例，再按概率随机选反射/折射
                    new_dir = ti.Vector([0.0, 0.0, 0.0])
                    if tir:
                        # 全反射：折射不存在，只能反射
                        new_dir = normalize(reflect(rd, outward_n))
                    else:
                        # Schlick 近似：r0 为垂直入射时的反射率，cos_x 为入射角余弦
                        cos_x = -rd.dot(outward_n)
                        r0 = ((1.0 - ior[None]) / (1.0 + ior[None])) ** 2
                        fresnel = r0 + (1.0 - r0) * (1.0 - cos_x) ** 5
                        if ti.random(ti.f32) < fresnel:
                            new_dir = normalize(reflect(rd, outward_n))
                        else:
                            new_dir = refr_dir

                    ro = p + new_dir * 1e-4    # 沿新方向偏移，避免自相交
                    rd = new_dir
                    throughput *= obj_color    # 玻璃的轻微吸收/染色
                    # 继续循环追踪

                # 分支 3：漫反射材质
                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)

                    # --- 硬阴影检测 ---
                    # 从当前交点向光源发射暗影射线，同样需要法线偏移
                    shadow_ray_orig = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)

                    # 判断：如果去光源的路上没被挡住 (或者遮挡物比光源还远)，则计算光照
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0  # 被挡住了！

                    # 简单的 Phong 光照 (由于是 Whitted-style，只算直接光)
                    ambient = 0.2 * obj_color
                    direct_light = ambient

                    # 如果不在阴影里，再额外加上漫反射的光
                    if in_shadow == 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse

                    # 将当前点的颜色乘以积累的能量，加到最终颜色里
                    final_color += throughput * direct_light

                    # 漫反射表面会打散光线，Whitted 风格下主射线到此终止
                    break

            pixel_color += final_color

        # 多个样本求平均后写入像素并做色调映射
        pixels[i, j] = ti.math.clamp(pixel_color / spp, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化光源位置、弹射次数、采样数与折射率
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 6   # 玻璃需要多次弹射（进入 + 内部 + 离开）
    num_samples[None] = 4   # 每像素采样数
    ior[None] = 1.5         # 典型玻璃折射率

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.32):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 10)
            num_samples[None] = gui.slider_int('Samples (AA)', num_samples[None], 1, 16)
            ior[None] = gui.slider_float('Glass IOR', ior[None], 1.0, 2.5)

        window.show()

if __name__ == '__main__':
    main()