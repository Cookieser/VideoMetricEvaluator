import torch
import numpy as np
from metrics import MetricEvaluator  


def generate_sample_videos(batch_size=2, time_steps=30, height=512, width=512):
    # [B, T, C, H, W]
    video1 = torch.randint(0, 256, (batch_size, time_steps, 3, height, width), dtype=torch.uint8).float()
    noise = torch.randn_like(video1) * 10  
    video2 = torch.clamp(video1 + noise, 0, 255)  
    return video1, video2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = MetricEvaluator(device)

videos1, videos2 = generate_sample_videos()

results = evaluator.evaluate(videos1, videos2, metrics=["psnr", "ssim", "lpips"])

print("Result")
for metric, value in results.items():
    print(f"{metric.upper()}: {value:.4f}")