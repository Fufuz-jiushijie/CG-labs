# 实验0：万有引力粒子系统
本实验实现了一个基于万有引力原理的粒子系统，模拟了粒子之间的引力作用和运动。用户可以通过鼠标拖动来改变粒子的位置，观察粒子之间的引力交互效果。
## 1.项目结构
```
/src/Work0/
├── config.py        # 配置文件，包含物理系统参数和渲染系统参数
├── main.py          # 主程序，包含初始化、更新和渲染循环
├── physics.py       # 物理模拟类，实现了万有引力计算和粒子运动更新
```

## 2. 环境管理
本项目使用uv进行全局环境管理，根目录下的uv.lock和pyproject.toml文件包含了项目的依赖信息。需要克隆整个项目以包含它们。
```
git clone https://github.com/Fufuz-jiushijie/CG-labs.git
uv sync
uv run -m src.Work0.main
```

## 3.运行效果
![这里写图片的描述文字](../../gifs/Work0.gif)