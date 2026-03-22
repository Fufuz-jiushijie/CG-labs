import taichi as ti
from src.Work2.class_renderstate import RenderState
from src.Work2.mvp import compute_transform

def main():
    ti.init(arch=ti.cpu)

    state = RenderState(shape_type="cube")
    
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=state.num_vertices)
    screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=state.num_vertices)
    
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
    
    gui = ti.GUI("3D Transformation (Taichi)", res=(700, 700))
    
    while gui.running:
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
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
        
        for i, j in state.edges:
            p1 = screen_coords[i]
            p2 = screen_coords[j]
            gui.line(p1, p2, radius=2, color=0xFFFFFF)
        
        gui.show()

if __name__ == '__main__':
    main()
