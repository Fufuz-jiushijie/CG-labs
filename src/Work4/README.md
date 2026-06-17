# 实验 4：Phong / Blinn-Phong 光照模型（Taichi GPU 光线追踪）

本实验在 **Taichi GPU** 上实现 **Whitted 风格的一次反弹光线追踪**：主射线与场景求交后计算局部光照（环境光 + 漫反射 + 高光），场景包含 **球体** 与 **圆锥**，并在 GGUI 中实时调节材质系数。

在基础 Phong 模型之上，本实验完成两项选做：

1. **Blinn-Phong 升级**：用半程向量 $\mathbf{H}$ 计算高光，替换 Phong 的反射向量 $\mathbf{R}$。
2. **硬阴影**：在交点向光源发射暗影射线，被遮挡的点只保留环境光。

---

## 1. 实验目标

| 目标 | 说明 |
|------|------|
| 射线—几何求交 | 球体、竖直圆锥（二次方程 + 高度范围裁剪） |
| Phong 模型 | `Ka` / `Kd` / `Ks` / `shininess` 四项可调 |
| Blinn-Phong 模型 | 计算半程向量 $\mathbf{H}$，可与 Phong 实时切换 |
| 硬阴影 | 暗影射线遮挡测试，被遮挡处仅有环境光 |
| 最近交点 | 多物体取最小正 `t` |
| 实时交互 | 滑块 + 开关调节材质与渲染模式 |

---

## 2. 光照模型原理

局部光照把一个着色点的颜色拆成三部分相加：**环境光 + 漫反射 + 高光**。

### 2.1 环境光（Ambient）

环境光是对“场景中四处弹射后均匀照亮一切的间接光”的粗暴近似，用一个常数项代替。它和入射角、视角都无关，只保证物体背光面不至于纯黑：

$$
I_{ambient} = K_a \cdot I_{light} \cdot C_{obj}
$$

### 2.2 漫反射（Diffuse）

理想漫反射面（Lambert 面）把入射光向各个方向均匀散射，亮度只取决于表面法线 $\mathbf{N}$ 与光源方向 $\mathbf{L}$ 的夹角。法线越正对光源越亮：

$$
I_{diffuse} = K_d \cdot \max(\mathbf{N}\cdot\mathbf{L},\,0) \cdot I_{light} \cdot C_{obj}
$$

取 `max(·, 0)` 是为了丢掉背光面（点积为负）的情况。

### 2.3 高光（Specular）—— Phong 模型

高光模拟光滑表面的镜面反射亮斑。Phong 的思路是：先求入射光关于法线的**反射向量** $\mathbf{R}$，再看它和**视线方向** $\mathbf{V}$ 有多接近。两者越对齐，说明越靠近“镜面正反射”的方向，高光越强：

$$
\mathbf{R} = 2(\mathbf{N}\cdot\mathbf{L})\mathbf{N} - \mathbf{L}
$$

$$
I_{specular} = K_s \cdot \max(\mathbf{R}\cdot\mathbf{V},\,0)^{\,n} \cdot I_{light}
$$

指数 $n$（shininess）控制高光的“锐利程度”：$n$ 越大，亮斑越小越集中，表面看起来越像抛光金属或塑料；$n$ 越小，高光越大越柔和。

### 2.4 高光升级 —— Blinn-Phong 模型

Phong 每个着色点都要算一次反射向量 $\mathbf{R}$。Blinn 提出了一个更省事也更稳的替代：不算反射向量，而是算光源方向和视线方向的**半程向量（half vector）** $\mathbf{H}$，它正好指向 $\mathbf{L}$ 和 $\mathbf{V}$ 的“正中间”：

$$
\mathbf{H} = \frac{\mathbf{L} + \mathbf{V}}{\|\mathbf{L} + \mathbf{V}\|}
$$

然后用 $\mathbf{H}$ 和法线 $\mathbf{N}$ 的接近程度衡量高光：当 $\mathbf{H}$ 与 $\mathbf{N}$ 重合时，说明此时正好满足镜面反射条件，高光最强：

$$
I_{specular} = K_s \cdot \max(\mathbf{N}\cdot\mathbf{H},\,0)^{\,n} \cdot I_{light}
$$

**两者在高光边缘、尤其大入射角下的差异：**

- Phong 用的是 $\mathbf{R}\cdot\mathbf{V}$，Blinn-Phong 用的是 $\mathbf{N}\cdot\mathbf{H}$。可以证明 $\mathbf{H}$ 与 $\mathbf{N}$ 的夹角大约是 $\mathbf{R}$ 与 $\mathbf{V}$ 夹角的一半，所以相同 $n$ 下 **Blinn-Phong 的高光会更大、更柔和**。要得到接近的视觉效果，Blinn-Phong 的指数通常要取 Phong 的 2～4 倍。
- **大入射角（掠射）时差异最明显**：当光源与视线都接近贴着表面时，Phong 的 $\mathbf{R}\cdot\mathbf{V}$ 容易掉到负值或接近 0，高光会被突然截断，亮斑边缘显得发硬、甚至出现不自然的断裂。而 Blinn-Phong 的 $\mathbf{N}\cdot\mathbf{H}$ 在掠射角下衰减得更平缓，高光被拉成一条柔和的长条，边缘过渡更自然——这也更符合真实粗糙表面在掠射时高光“拉长”的现象。
- 结论：Blinn-Phong 计算更简单（省去反射向量），在大入射角下边缘表现更稳定、更接近真实，因此成为后来固定管线（OpenGL/DirectX）默认的高光模型。

### 2.5 硬阴影（Hard Shadow）

判断一个着色点是否在阴影里，思路很直接：从交点朝光源方向发射一条**暗影射线（Shadow Ray）**，看它在抵达光源之前会不会撞到别的物体。

- 如果中途被挡住 → 该点照不到这盏灯，**只保留环境光**，漫反射和高光都置零；
- 如果一路畅通 → 正常计算完整光照。

两个实现细节：

- **自遮挡偏移**：交点本身就在物体表面上，暗影射线若从交点原地出发，会因浮点误差立刻和自己相交，产生满屏黑斑（shadow acne）。解决办法是把射线起点沿法线方向推出一个小量 $\varepsilon$ 再发射。
- **距离裁剪**：遮挡只在“交点到光源之间”才算数，所以命中距离 `t` 必须满足 `t < dist_to_light`；光源后面的物体不构成遮挡。

这种方式得到的是边缘锐利的“硬阴影”，因为光源被当作一个理想点，每个点要么完全可见、要么完全被挡，没有中间过渡（软阴影需要面光源 + 多次采样）。

---

## 3. 场景与光照

分辨率 **800×600**。摄像机 `(0, 0, 5)`，主射线方向由像素坐标生成。

| 物体 | 几何 | 基色 |
|------|------|------|
| 左球 | 中心 `(-1.2, -0.2, 0)`，半径 1.2 | 红色 |
| 右圆锥 | 顶点 `(1.2, 1.2, 0)`，底面 `y=-1.4`，底半径 1.2 | 紫色 |
| 背景 | 未击中 | 深青 `(0.05, 0.15, 0.15)` |

固定点光源 `(2, 3, 4)`，白光。球与圆锥相互投射阴影。

---

## 4. 项目结构

```
src/Work4/
├── main.py      # 求交、Phong/Blinn-Phong 着色、硬阴影、GGUI
└── README.md
```

---

## 5. 环境与运行

```bash
uv sync
uv run -m src.Work4.main
```

### 操作说明（Material Parameters）

| 控件 | 含义 |
|------|------|
| Ka (Ambient) | 环境光系数 |
| Kd (Diffuse) | 漫反射系数 |
| Ks (Specular) | 镜面反射系数 |
| N (Shininess) | 高光指数（1–128） |
| Blinn-Phong | 勾选用 Blinn-Phong，取消则退回 Phong |
| Hard Shadow | 勾选开启硬阴影 |

> 建议对比方式：固定 `N`，反复勾选 / 取消 **Blinn-Phong**，观察高光大小与边缘；再把球转到光源侧面（掠射角）感受边缘差异。

---

## 6. 关键代码实现

### 6.1 Blinn-Phong / Phong 切换

高光部分按开关二选一：Blinn-Phong 走半程向量 $\mathbf{H}$，Phong 走反射向量 $\mathbf{R}$。

```176:184:src/Work4/main.py
                # 高光：Blinn-Phong 用半程向量 H，Phong 用反射向量 R
                spec = 0.0
                if use_blinn[None] == 1:
                    H = normalize(L + V)
                    spec = ti.max(0.0, N.dot(H)) ** shininess[None]
                else:
                    R = normalize(reflect(-L, N))
                    spec = ti.max(0.0, R.dot(V)) ** shininess[None]
                specular = Ks[None] * spec * light_color
```

### 6.2 暗影射线遮挡测试

把场景几何抽成一个共用函数 `occluded`，让主射线和暗影射线用同一套求交逻辑。命中距离限制在 `1e-3 < t < max_dist`：下界 `1e-3` 配合起点偏移避免自遮挡，上界 `max_dist`（交点到光源距离）保证只统计中间的遮挡物。

```106:116:src/Work4/main.py
@ti.func
def occluded(ro, rd, max_dist):
    """暗影射线：从 ro 沿 rd 出发，在 max_dist 之内若击中任意物体则返回 True"""
    blocked = False
    t_sph, _ = intersect_sphere(ro, rd, SPH_CENTER, SPH_RADIUS)
    if 1e-3 < t_sph < max_dist:
        blocked = True
    t_cone, _ = intersect_cone(ro, rd, CONE_APEX, CONE_BASE_Y, CONE_RADIUS)
    if 1e-3 < t_cone < max_dist:
        blocked = True
    return blocked
```

### 6.3 着色主流程

环境光始终累加；若开启阴影且被遮挡，则跳过漫反射与高光。

```160:186:src/Work4/main.py
            # 环境光始终存在
            ambient = Ka[None] * light_color * hit_color

            # --- 硬阴影：沿法线偏移交点后发射暗影射线探测光源 ---
            in_shadow = False
            if enable_shadow[None] == 1:
                dist_to_light = (light_pos - p).norm()
                shadow_origin = p + N * 1e-3
                in_shadow = occluded(shadow_origin, L, dist_to_light)

            color = ambient
            if not in_shadow:
                # 漫反射
                diff = ti.max(0.0, N.dot(L))
                diffuse = Kd[None] * diff * light_color * hit_color
                ...
                color = ambient + diffuse + specular
```

---

## 7. 效果展示与结果解读

### 7.1 调节 Phong 材质参数

<div align="center">
<img src="../../gifs/Work4/Phong光照模型调整参数.gif" alt="Phong 光照：调节 Ka、Kd、Ks 与高光指数" width="520">
</div>

拖动四个滑块可以看到每一项的独立作用：

- **Ka 增大**：球和圆锥整体被均匀提亮，连背光面也跟着变亮，但画面随之发灰、失去立体感——这正是环境光“无方向、压低对比”的特征。
- **Kd 增大**：朝向光源的一侧明显变亮、背光侧仍暗，明暗过渡自然，物体的体积感最依赖这一项。
- **Ks 增大**：高光亮斑变得更亮更扎眼，但只影响那一小块镜面反射区域，其余部分不变。
- **N（shininess）增大**：高光亮斑收缩、变锐，表面看起来更像抛光塑料 / 金属；N 减小则亮斑摊开、变得柔和。

### 7.2 Blinn-Phong vs Phong（选做 1）

固定 `N=32`、`Ks=0.5`，并开启硬阴影，反复切换 **Blinn-Phong** 开关：

<div align="center">
<img src="../../gifs/Work4/Blinn-Phong_vs_Phong.gif" alt="相同 shininess 下 Blinn-Phong 与 Phong 的高光对比" width="520">
</div>

- **相同指数下高光大小不同**：勾选 Blinn-Phong 时，球面的高光亮斑明显**更大、更柔和**；取消（退回 Phong）后亮斑**收缩、变锐、更集中**。原因正如原理所述——`H·N` 的夹角约为 `R·V` 的一半，所以同一个 `N` 下 Blinn-Phong 等效更“钝”，要追平视觉效果需把指数调到 Phong 的 2~4 倍。
- **高光边缘的表现**：在球体靠近边缘、光线接近掠射的区域，Phong 的亮斑边缘更硬、收尾更突兀；Blinn-Phong 的高光衰减更平缓，边缘过渡更自然，不会出现被突然“切掉”的感觉。这与大入射角下 `R·V` 容易掉到 0 附近、而 `N·H` 衰减更温和的分析一致。
- **稳定性**：整个切换过程画面无明显跳变，说明两套高光只是同一着色框架下的不同公式，可以无缝替换。

### 7.3 硬阴影（选做 2）

两段 gif 中均已开启 **Hard Shadow**：可以看到红球与圆锥朝光面正常受光、背光侧变暗，而各自背向光源 `(2,3,4)` 的区域、以及物体相互遮挡之处出现了**边缘锐利的暗影**。被暗影射线判定为遮挡的像素只保留环境光，因此呈现出明显比漫反射暗、但又不是纯黑的色调——这正是“只算 Ambient”的直接结果。阴影边缘干净利落、没有半影过渡，体现了点光源硬阴影的特点。

---

## 8. 与课程知识点的对应

| 知识点 | 本仓库实现 |
|--------|------------|
| 射线—曲面求交 | `intersect_sphere` / `intersect_cone` |
| Phong 高光 | `R.dot(V) ** shininess`（反射向量） |
| Blinn-Phong 高光 | `N.dot(H) ** shininess`（半程向量 `H = normalize(L+V)`） |
| 环境光 / 漫反射 | `render()` 内核中 ambient + diffuse |
| 硬阴影 | `occluded()` 暗影射线 + 法线偏移 + 距离裁剪 |
| GPU 逐像素渲染 | `@ti.kernel def render()` |

---

## 9. 参考文献

- Phong, *Illumination for Computer Generated Pictures* (1975)
- Blinn, *Models of Light Reflection for Computer Synthesized Pictures* (1977)
- [Taichi 文档](https://docs.taichi-lang.org/)

---
