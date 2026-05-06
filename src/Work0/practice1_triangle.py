import taichi as ti

ti.init(arch=ti.gpu)

vertices         = ti.Vector.field(2, ti.f32, shape=50)
indices          = ti.field(ti.i32, shape=50 * 3)
per_vertex_color = ti.Vector.field(3, ti.f32, shape=50)

color = (0.2, 0.2, 0.2) # 把背景调暗一点，更好看

@ti.kernel
def init_data():
    # 1. 随机初始化 200 个顶点的坐标和颜色
    for i in range(50):
        # ti.random() 默认生成 0.0 到 1.0 的浮点数
        vertices[i] = ti.Vector([ti.random(), ti.random()])
        per_vertex_color[i] = ti.Vector([i / 50.0, 1.0 - i / 50.0, 0.2])
        
    # 2. 随机生成 600 个连线索引 (拼成 200 个三角形)
    for i in range(50):
        # int(ti.random() * 200) 保证生成 0~199 之间的安全整数
        indices[i * 3 + 0] = int(ti.random() * 50)
        indices[i * 3 + 1] = int(ti.random() * 50)
        indices[i * 3 + 2] = int(ti.random() * 50)

def main():
    res = (640, 480)
    window = ti.ui.Window(name="Taichi Random Triangles", res=res, fps_limit=60, pos=(200,200))
    
    # 【核心改动】：把数据初始化放到死循环外面，只执行一次！
    init_data()
    
    while window.running:
        canvas = window.get_canvas()
        canvas.set_background_color(color)
        
        # 每帧只负责把算好的数据画出来
        canvas.triangles(vertices, color=(1.0, 1.0, 0.0), indices=indices, per_vertex_color=per_vertex_color)
        
        window.show()

if __name__ == "__main__":
    main()