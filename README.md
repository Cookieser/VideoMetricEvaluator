# 📊 Video Quality Metrics Evaluator

This project provides a unified interface `MetricEvaluator` for evaluating multiple quality metrics between two videos, including:

- **PSNR**
- **SSIM**
- **LPIPS**
- **FVD**
- **TLP100**（Temporal LPIPS）

It supports full video processing in `[B, T, C, H, W]` format and automatically handles frame-wise or global evaluations.

------

## 🚀 Quick Start

The following demo shows how to load a video, add noise, and evaluate quality differences between two videos:

### ✅ Example: `demo.py`

```
import torch
import numpy as np
from metrics import MetricEvaluator  
import cv2

video_path = './test.mp4'

# Load video and convert to PyTorch tensor
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

# Add noise to create a comparison video
noise = torch.randn_like(video_tensor)
noisy_video = torch.clamp(video_tensor + noise, 0, 255)

# Initialize evaluator and compute metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = MetricEvaluator(device)
results = evaluator.evaluate(video_tensor, noisy_video, metrics=["psnr", "ssim", "lpips", "fvd", "tlp"])

# Print results
print("Result:")
for metric, value in results.items():
    print(f"{metric.upper()}: {value:.4f}")
```
![image-20250405111134175](https://pic-1306483575.cos.ap-nanjing.myqcloud.com/image-20250405111134175.png)
------

## 📦 Dependencies

Install required packages:

```
pip install opencv-python torch torchvision lpips scikit-image
```

------

## 🧠 Supported Metrics

| Metric    | Description                                                         |
| ------- | ------------------------------------------------------------ |
| `psnr`  | Peak Signal-to-Noise Ratio. Measures frame-level pixel fidelity.                               |
| `ssim`  | Structural Similarity Index. Measures image structural similarity.                           |
| `lpips` | Learned Perceptual Image Patch Similarity. Uses deep networks to evaluate perceptual differences.                       |
| `fvd`   | Frechet Video Distance. Measures overall video quality difference (requires I3D).     |
| `tlp`   | Temporal LPIPS. Evaluates temporal consistency, inspired by [TecoGAN](https://arxiv.org/abs/1811.09393) |

------

## 📁 Project Structure

```
project/
├── demo.py                # Example script
├── metrics.py             # MetricEvaluator implementation
├── test.mp4               # Sample video (user provided)
└── fvd/
    └── styleganv/         # FVD dependency modules
```

------

## 📌 Notes

- All input videos must be in [B, T, C, H, W] format with pixel values in [0, 255].
- LPIPS and FVD use alex and styleganv backbones by default. Modify parameters to change.
- For FVD evaluation, each clip must have at least T ≥ 10 frames.
