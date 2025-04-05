# 📊 Video Quality Metrics Evaluator

本项目提供一个统一的接口 `MetricEvaluator` 来评估两个视频之间的多种质量指标，包括：

- **PSNR**
- **SSIM**
- **LPIPS**
- **FVD**
- **TLP100**（Temporal LPIPS）

支持处理整段视频 `[B, T, C, H, W]` 格式，自动执行逐帧或全局评估。

------

## 🚀 快速上手

下面的 DEMO 展示了如何加载一个视频，添加噪声并评估两段视频的质量差异：

### ✅ 示例：`demo.py`

```
import torch
import numpy as np
from metrics import MetricEvaluator  
import cv2

video_path = './test.mp4'

# 读取视频并转换为 PyTorch 张量
cap = cv2.VideoCapture(video_path)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

video_np = np.stack(frames, axis=0)  # [T, H, W, C]
video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float().unsqueeze(0)  # [1, T, C, H, W]

# 添加噪声生成对比视频
noise = torch.randn_like(video_tensor)
noisy_video = torch.clamp(video_tensor + noise, 0, 255)

# 初始化评估器并计算指标
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = MetricEvaluator(device)
results = evaluator.evaluate(video_tensor, noisy_video, metrics=["psnr", "ssim", "lpips", "fvd", "tlp"])

# 打印结果
print("Result:")
for metric, value in results.items():
    print(f"{metric.upper()}: {value:.4f}")
```
![image-20250405111134175](https://pic-1306483575.cos.ap-nanjing.myqcloud.com/image-20250405111134175.png)
------

## 📦 依赖项

可以通过以下方式安装依赖：

```
pip install opencv-python torch torchvision lpips scikit-image
```

------

## 🧠 支持的指标说明

| 指标    | 描述                                                         |
| ------- | ------------------------------------------------------------ |
| `psnr`  | 峰值信噪比，衡量帧级像素保真度                               |
| `ssim`  | 结构相似度，用于衡量图像结构相似性                           |
| `lpips` | 感知距离指标，使用深度网络评估感知差异                       |
| `fvd`   | Frechet Video Distance，用于整体视频质量对比（依赖 I3D）     |
| `tlp`   | Temporal LPIPS，对时间一致性进行衡量，参考 [TecoGAN](https://arxiv.org/abs/1811.09393) |

------

## 📁 项目结构

```
project/
├── demo.py                # 使用示例
├── metrics.py             # MetricEvaluator 实现
├── test.mp4               # 示例视频（用户提供）
└── fvd/
    └── styleganv/         # FVD 计算依赖的模块
```

------

## 📌 注意事项

- 所有视频输入需为 `[B, T, C, H, W]` 格式，像素值范围 `[0, 255]`。
- LPIPS 和 FVD 默认使用 `alex` 和 `styleganv` 版本，如需更换请修改参数。
- FVD 评估需要每段 clip 的时间帧数 `T ≥ 10`。
