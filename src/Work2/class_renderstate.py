import taichi as ti

class RenderState:
    """
    负责在纯 Python 层接管所有的默认参数和 GUI 交互状态。
    绕过 Taichi Kernel 不支持默认参数的限制，方便后续扩展 UI 控制。
    """
    def __init__(self, shape_type: str = "cube"):
        self.shape_type = shape_type
        
        if shape_type == "triangle":
            self.num_vertices = 3
            self.edges = [(0, 1), (1, 2), (2, 0)]
        elif shape_type == "cube":
            self.num_vertices = 8
            self.edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")
        
        # 视图参数 (View)
        self.eye_pos = ti.math.vec3([0.0, 0.0, 5.0])
        self.target_pos = ti.math.vec3([0.0, 0.0, -1.0])
        self.up = ti.math.vec3([0.0, 1.0, 0.0])
        
        # 模型参数 (Model)
        self.translation = ti.math.vec3([0.0, 0.0, 0.0])
        self.rotation_angles = ti.math.vec3([0.0, 0.0, 0.0])
        self.scale = ti.math.vec3([1.0, 1.0, 1.0])
        
        # 投影参数 (Projection)
        self.fov_y = 45.0 * ti.math.pi / 180.0
        self.aspect_ratio = 1.0
        self.z_near = 0.1
        self.z_far = 50.0