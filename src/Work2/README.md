# CG实验2：

### Taichi行为

请记住这条 Taichi 编程的黄金法则：
宏观的、遍历整个数据场（Field）的任务 -> 交给 @ti.kernel（程序的入口）。
微观的、只针对单个元素/粒子的数学计算 -> 交给 @ti.func（代码的零件）。

一般来说，一个优秀的 Taichi 程序，在每一帧/每一步的物理模拟中，从 Python 端发起的 Kernel 调用次数应该尽可能少（最好是个位数），而大量的复杂逻辑都应该被封装成 Func 在 Kernel 内部消化掉。

@ti.func 是给taichi静态编译的函数，不通过python解释器调用。

一旦加上了 @ti.kernel，里面的每一句运行时代码都必须是 Taichi 能理解的（即 ti.math、ti.Vector 等自带的运算），千万不要让 GPU 去尝试理解 Python 的 list、dict 或者是 numpy 数组。