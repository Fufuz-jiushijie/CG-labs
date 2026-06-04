# %% [markdown]
# ## Work6：从 `output_meshes` 生成「球 → 牛」形变 GIF
#
# ### ModelScope 必读（避免 No module named 'pytorch3d'）
#
# 1. **与跑 `main.py` 必须在同一 Notebook、同一内核**（菜单「内核 → 重启」后需重新 `%pip` 安装）。
# 2. 安装请用 **`%pip`**，不要用 `!pip`（后者常装到别的 Python，内核仍找不到包）。
# 3. 若训练用的是 **`!python main.py`**，包装在子进程里，**当前内核可能没有 pytorch3d**；
#    请在本 Notebook 先跑下面「环境检查」单元，或改用 `%run main.py` / 单元格内直接跑 main。
# 4. 生成 GIF 前建议先执行：`import pytorch3d` 无报错，再 `%run make_mesh_gif.py`。
#
# ### 环境检查（复制到独立单元格，应先于本脚本运行）
# ```python
# import sys
# print("内核 Python:", sys.executable)
# import torch
# print("torch", torch.__version__, "cuda", torch.cuda.is_available())
# import pytorch3d
# print("pytorch3d", pytorch3d.__version__)
# ```
#
# ### 若检查失败，在本 Notebook 用 %pip 安装（约 5–10 分钟）
# ```python
# %pip install --upgrade pip
# %pip install fvcore iopath matplotlib ninja pillow
# %pip install "git+https://gitee.com/hongwenzhang/pytorch3d.git" --no-build-isolation
# ```
# 安装完成后**不要重启内核**，直接再跑环境检查 → `%run make_mesh_gif.py`。

# %%
import glob
import os
import re
import subprocess
import sys

import numpy as np
import torch
from IPython.display import Image, display
from PIL import Image as PILImage

# 设为 True 时：检测到缺失则对「当前内核」自动 pip（与 main 相同的 Gitee 源）
AUTO_INSTALL_IF_MISSING = False

# ---------- 可按 ModelScope 工作目录修改 ----------
INPUT_DIR = "output_meshes"
OUTPUT_GIF = "Cow_mesh.gif"
IMAGE_SIZE = 512
FPS = 8
FRAME_EVERY = 1
MAX_FRAMES = None
CAM_DIST = 2.7
CAM_ELEV = 20.0
CAM_AZIM = 45.0
# -----------------------------------------------

_INSTALL_HELP = """
未在当前 Jupyter 内核中找到 pytorch3d。

常见原因：
  • 用了 !pip / !python main.py，包装未装进本内核；
  • 重启过内核，但未重新 %pip 安装；
  • 在另一个 Notebook 里装的包。

请在本 Notebook 新建单元格执行（务必用 %pip，不要用 !pip）：

  %pip install --upgrade pip
  %pip install fvcore iopath matplotlib ninja pillow
  %pip install "git+https://gitee.com/hongwenzhang/pytorch3d.git" --no-build-isolation

然后运行环境检查（import pytorch3d），再 %run make_mesh_gif.py。
或将本文件顶部 AUTO_INSTALL_IF_MISSING = True 后重新运行（会对当前内核自动安装）。
"""


def _pip_to_current_kernel(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print("执行:", " ".join(cmd))
    subprocess.check_call(cmd)


def ensure_pytorch3d():
    try:
        import pytorch3d

        print(f"pytorch3d {pytorch3d.__version__}  @  {sys.executable}")
        return pytorch3d
    except ImportError:
        print(_INSTALL_HELP)
        print("当前内核 Python:", sys.executable)
        if not AUTO_INSTALL_IF_MISSING:
            raise
        print("\n[AUTO_INSTALL_IF_MISSING] 正在向当前内核安装，请耐心等待…")
        _pip_to_current_kernel("install", "--upgrade", "pip")
        _pip_to_current_kernel(
            "install", "fvcore", "iopath", "matplotlib", "ninja", "pillow"
        )
        _pip_to_current_kernel(
            "install",
            "git+https://gitee.com/hongwenzhang/pytorch3d.git",
            "--no-build-isolation",
        )
        import pytorch3d

        print(f"安装完成: pytorch3d {pytorch3d.__version__}")
        return pytorch3d


ensure_pytorch3d()

from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

# main.py 的 save_obj 只存几何，无材质；Phong 着色必须有 textures
MESH_COLOR = (0.78, 0.74, 0.70)


def parse_epoch(path: str) -> int:
    m = re.search(r"mesh_epoch_(\d+)\.obj$", os.path.basename(path))
    if not m:
        raise ValueError(f"无法解析 epoch: {path}")
    return int(m.group(1))


def list_objs(folder: str) -> list[str]:
    paths = glob.glob(os.path.join(folder, "mesh_epoch_*.obj"))
    if not paths:
        raise FileNotFoundError(
            f"在 {os.path.abspath(folder)} 未找到 mesh_epoch_*.obj。\n"
            "请先运行 main.py，或将 output_meshes 上传到本 notebook 工作目录。"
        )
    return sorted(paths, key=parse_epoch)


def subsample(paths: list[str], every: int, max_frames: int | None) -> list[str]:
    if every > 1:
        paths = paths[::every]
    if max_frames is not None and len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
        paths = [paths[i] for i in idx]
    return paths


def build_renderer(image_size: int, device: torch.device) -> MeshRenderer:
    R, T = look_at_view_transform(dist=CAM_DIST, elev=CAM_ELEV, azim=CAM_AZIM)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )


def load_mesh(path: str, device: torch.device) -> Meshes:
    verts, faces, _ = load_obj(path)
    verts = verts.to(device)
    faces_idx = faces.verts_idx.to(device)
    color = torch.tensor(MESH_COLOR, dtype=torch.float32, device=device)
    verts_rgb = color.expand(1, verts.shape[0], 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    return Meshes(verts=[verts], faces=[faces_idx], textures=textures)


def render_rgb(mesh: Meshes, renderer: MeshRenderer) -> np.ndarray:
    with torch.no_grad():
        img = renderer(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


def make_gif(
    input_dir: str = INPUT_DIR,
    output_gif: str = OUTPUT_GIF,
    image_size: int = IMAGE_SIZE,
    fps: float = FPS,
    frame_every: int = FRAME_EVERY,
    max_frames: int | None = MAX_FRAMES,
    show_inline: bool = True,
) -> str:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    paths = subsample(list_objs(input_dir), frame_every, max_frames)
    renderer = build_renderer(image_size, device)

    frames: list[np.ndarray] = []
    for k, path in enumerate(paths):
        epoch = parse_epoch(path)
        rgb = render_rgb(load_mesh(path, device), renderer)
        frames.append(rgb)
        print(f"[{k + 1}/{len(paths)}] epoch {epoch:03d}  {os.path.basename(path)}")

    duration_ms = int(1000 / max(fps, 0.1))
    pil = [PILImage.fromarray(f) for f in frames]
    out_path = os.path.abspath(output_gif)
    pil[0].save(
        out_path,
        save_all=True,
        append_images=pil[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"已保存: {out_path}（{len(frames)} 帧, {fps} fps）")

    if show_inline:
        display(Image(filename=out_path))

    return out_path


if __name__ == "__main__":
    make_gif()
