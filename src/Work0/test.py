import taichi as ti
import taichi.math as tm
from taichi.types import int16

ti.init(arch=ti.gpu)

v = ti.Vector.field(n=3, dtype=ti.f32, shape=(640, 480))
print(v[0, 1])
@ti.kernel
def printf(mode: int16):
    for i,j in v:
        # clear background every frame
        if i%20 == 0 and j%20 == 0:
            if mode == 1:
                v[i, j] = tm.vec3(100, 0, 0)
            elif mode == 2:
                v[i, j] = tm.vec3(0,100,0)
            elif mode == 3:
                v[i, j] = tm.vec3(0,0,100)
        if mode == 4:
            v[i, j] = tm.vec3(1, 1, 1)

            # print(i, " ",  j)

gui = ti.GUI("fuck", res=(640, 480))
while gui.running:
    gui.set_image(v)
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == '1':
            printf(1)
        elif gui.event.key == '2':
            printf(2)
        elif gui.event.key == '3':
            printf(3)
        elif gui.event.key == '4':
            printf(4)
    gui.show()

