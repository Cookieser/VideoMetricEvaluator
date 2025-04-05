import torch
import numpy as np
from metrics import MetricEvaluator  
import cv2
import torch
import numpy as np

# def generate_sample_videos(batch_size=2, time_steps=30, height=512, width=512):
#     # [B, T, C, H, W]
#     video1 = torch.randint(0, 256, (batch_size, time_steps, 3, height, width), dtype=torch.uint8).float()
#     noise = torch.randn_like(video1) * 10  
#     video2 = torch.clamp(video1 + noise, 0, 255)  
#     return video1, video2


video_path = './test.mp4'

cap = cv2.VideoCapture(video_path)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    frames.append(frame)

cap.release()

video_np = np.stack(frames, axis=0)  # T x H x W x C

# PyTorch tensor: (T, C, H, W)
video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()

# (B, T, C, H, W)
video_tensor = video_tensor.unsqueeze(0)

noise = torch.randn_like(video_tensor) 
noisy_video = torch.clamp(video_tensor + noise, 0, 255)

print(video_tensor.shape)
print(noisy_video.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = MetricEvaluator(device)

videos1, videos2 = video_tensor,noisy_video

results = evaluator.evaluate(videos1, videos2, metrics=["psnr", "ssim", "lpips"])

print("Result:")
for metric, value in results.items():
    print(f"{metric.upper()}: {value:.4f}")