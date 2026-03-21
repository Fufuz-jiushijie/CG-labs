import taichi as ti


ti.init(arch=ti.cpu)

v = ti.Vector.field(n=3, dtype=ti.int16, shape=3)

@ti.kernel
def printf():
    for i in v:
        print(i)

printf()