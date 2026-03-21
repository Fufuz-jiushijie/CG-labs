import taichi as ti
# 在 main.py 中把原来的 import 改成这样：
from src.Work2.class_renderstate import RenderState
# 如果还有 renderer 模块，也一样：
from src.Work2.mvp import compute_transform
def main():
    # 初始化 Taichi，指定使用 CPU 后端
    ti.init(arch=ti.cpu)

    # 声明 Taichi 的 Field 来存储顶点和转换后的屏幕坐标
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
    screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)
    
    # 初始化三角形顶点
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]
    
    # 实例化状态控制器
    state = RenderState()
    
    gui = ti.GUI("3D Transformation (Taichi)", res=(700, 700))
    
    while gui.running:
        # 1. 处理 GUI 事件，实时更新 Python 层的状态
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            # 用 ad 控制三角形绕 Z 轴旋转为例
            elif gui.event.key == 'a':
                state.rotation_angles[2] += 0.1  # 弧度
            elif gui.event.key == 'd':
                state.rotation_angles[2] -= 0.1
            # # 控制相机 Z 轴推拉
            # elif gui.event.key == 'w':
            #     state.eye_pos[2] -= 0.5
            # elif gui.event.key == 's':
            #     state.eye_pos[2] += 0.5
        
        # 2. 将状态解包，作为参数一次性发射给 Taichi Kernel
        compute_transform(
            vertices,
            screen_coords,
            state.eye_pos, state.target_pos, state.up,
            state.translation, state.rotation_angles, state.scale,
            state.fov_y, state.aspect_ratio, state.z_near, state.z_far
        )
        
        # 3. 渲染绘制
        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]
        
        gui.line(a, b, radius=2, color=0xFF0000)
        gui.line(b, c, radius=2, color=0x00FF00)
        gui.line(c, a, radius=2, color=0x0000FF)
        
        gui.show()

if __name__ == '__main__':
    main()