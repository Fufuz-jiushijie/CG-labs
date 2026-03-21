# type: ignore
"""
    src/Work2/mvp.py
    计算内核    
"""

import taichi as ti

@ti.func
def get_model_matrix(translation: ti.math.vec3, rotation_angles: ti.math.vec3, scale: ti.math.vec3) -> ti.math.mat4: # type: ignore
    """
    参数:
        translation (ti.math.vec3): 平移向量 (tx, ty, tz)
        rotation_angles (ti.math.vec3): 绕 X, Y, Z 轴旋转的欧拉角
        scale (ti.math.vec3): 缩放向量 (sx, sy, sz)。
    返回:
        模型变换矩阵: T * Rz * Ry * Rx * S    
    注意: 
        rotation_angles 传入弧度制 (rad)
    """
    
    # 旋转角度
    rx = rotation_angles[0]
    ry = rotation_angles[1]
    rz = rotation_angles[2]
    
    # 计算三角函数
    cx, sx = ti.cos(rx), ti.sin(rx)
    cy, sy = ti.cos(ry), ti.sin(ry)
    cz, sz = ti.cos(rz), ti.sin(rz)
    
    # 绕 X 轴旋转矩阵
    Rx = ti.math.mat4([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  cx, -sx, 0.0],
        [0.0,  sx,  cx, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 绕 Y 轴旋转矩阵
    Ry = ti.math.mat4([
        [ cy, 0.0,  sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0,  cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 绕 Z 轴旋转矩阵
    Rz = ti.math.mat4([
        [ cz, -sz, 0.0, 0.0],
        [ sz,  cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 缩放矩阵
    S = ti.math.mat4([
        [scale[0], 0.0,      0.0,      0.0],
        [0.0,      scale[1], 0.0,      0.0],
        [0.0,      0.0,      scale[2], 0.0],
        [0.0,      0.0,      0.0,      1.0]
    ])
    
    # 平移矩阵
    T = ti.math.mat4([
        [1.0, 0.0, 0.0, translation[0]],
        [0.0, 1.0, 0.0, translation[1]],
        [0.0, 0.0, 1.0, translation[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 先缩放，再按 X->Y->Z 旋转，最后平移
    model_matrix = T @ Rz @ Ry @ Rx @ S
    
    return model_matrix


@ti.func
def get_view_matrix(eye_pos: ti.math.vec3, target_pos: ti.math.vec3, up: ti.math.vec3) -> ti.math.mat4: # type: ignore
    """
    视图变换矩阵 (View Matrix)：将世界坐标系转化为以相机为原点的局部坐标系。
    
    参数:
        eye_pos (ti.math.vec3): 相机在世界坐标系中的绝对位置 (Position)。
        target_pos (ti.math.vec3): 相机正在看的目标点位置 (LookAt)。
                                   (注：用 target_pos 减去 eye_pos 即可算出相机的前向向量 Forward/W)
        up_dir (ti.math.vec3): 相机的正上方参考朝向 (Up)，用于叉乘计算出右向向量 (Right/U) 和真实的上方向量 (True Up/V)。
        
    返回:
        ti.math.mat4: 4x4 的视图变换矩阵 (V = R_view * T_view)。
    """
    
    T_view = ti.math.mat4([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    w = ti.math.normalize((eye_pos - target_pos))
    u = ti.math.cross(up, w)
    v = ti.math.cross(w, u)  
    
    R_view = ti.math.mat4([
        [u[0], u[1], u[2], 0.0],
        [v[0], v[1], v[2], 0.0],
        [w[0], w[1], w[2], 0.0],
        [0.0,  0.0,  0.0,  1.0]
    ])
    
    V = R_view@T_view
    
    return V


@ti.func
def get_projection_matrix(fov_y: ti.f32, aspect_ratio: ti.f32, z_near: ti.f32, z_far: ti.f32) -> ti.math.mat4: # type: ignore
    """
    透视投影矩阵 (Perspective Projection Matrix):
        将视锥体空间挤压并映射到标准的坐标 (NDC, 即[-1,1]^3 的正方体)。
        
    核心思想：利用齐次坐标的 W 分量储存深度 Z, 配合硬件的透视除法实现“近大远小”，
        并用非线性方程保留深度相对顺序。
    
    参数:
        fov_y (ti.f32): 垂直视场角 (Field of View Y) 以弧度输入。
            角度越大，视野越广，物体显得越小。
        aspect_ratio (ti.f32): 屏幕宽高比 (Width / Height),
            用于结合 fov_y 确定水平方向的视野边界。
        z_near (ti.f32): 近裁剪面到相机的距离(必须大于0)。比这更近的物体会被剔除。
        z_far (ti.f32): 远裁剪面到相机的距离。比这更远的物体会被剔除。
        
    返回:
        ti.math.mat4: 4x4 的透视投影矩阵。
    """
    
    # 利用fov_y, near 和aspect_ratio计算出近截面的边界
    
    max_y = z_near * ti.tan(fov_y)
    min_y = -max_y
    max_x = max_y * aspect_ratio
    min_x = -max_x
    
    # 伸缩平移矩阵M_ortho = M_scale @ M_trans
    # 平移
    M_trans = ti.math.mat4([
        [1.0, 0.0, 0.0, -(max_x + min_x)/2.0],
        [0.0, 1.0, 0.0, -(max_y + min_y)/2.0],
        [0.0, 0.0, 1.0, -(z_far + z_near)/2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # 伸缩
    M_scale = ti.math.mat4([
        [2.0/(max_x - min_x), 0.0, 0.0, 0.0],
        [0.0, 2.0/(max_y - min_y), 0.0, 0.0],
        [0.0, 0.0, 2.0/(z_far - z_near),0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    
    M_proj_ortho = ti.math.mat4([
        [z_near, 0.0, 0.0, 0.0],
        [0.0, z_near, 0.0, 0.0],
        [0.0, 0.0, z_far+z_near, -z_far*z_near],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    return M_scale @ M_trans @ M_proj_ortho



@ti.kernel
def compute_transform(
    vertices: ti.template(),        # 传入：模型原始顶点的 field
    screen_coords: ti.template(),   # 传出：计算后写回屏幕坐标的 field
    
    # 变换参数
    eye_pos: ti.math.vec3,
    target_pos: ti.math.vec3,
    up: ti.math.vec3,
    translation: ti.math.vec3,
    rotation_angles: ti.math.vec3,
    scale: ti.math.vec3,
    fov_y: ti.f32,
    aspect_ratio: ti.f32,
    z_near: ti.f32,
    z_far: ti.f32
):
    """
    根据变换参数，更新顶点位置
    """
    model = get_model_matrix(translation, rotation_angles, scale)
    view = get_view_matrix(eye_pos, target_pos, up)
    proj = get_projection_matrix(fov_y, aspect_ratio, z_near, z_far)
    
    mvp = proj @ view @ model
    
    # 直接遍历输入的 ti.field 因为ti.field本身是一个可迭代对象
    for i in vertices: 
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换，自动写入对应索引的 screen_coords
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0
        
        # if i==0:
        #     print("A:")
        # elif i==1:
        #     print("B:")
        # else:
        #     print("C:")
        # print(screen_coords[i])


