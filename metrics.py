import numpy as np
import cv2
import lpips
import torch
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings("ignore")

class MetricEvaluator:
    def __init__(self, device):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex', pretrained=True).to(device)


    def compute_psnr(self, ref, dist):
        """
        ref, dist: np.ndarray with shape (B, C, H, W), values in [0, 255]
        Returns: PSNR mean value
        """
        assert ref.shape == dist.shape, "Input shape mismatch"
        
        ref = ref.astype(np.float32)
        dist = dist.astype(np.float32)
        
        mse = np.mean((ref - dist) ** 2, axis=(-2, -1))  # shape: (B, C)
        
        # Avoid log(0): if mse == 0, set PSNR = inf
        with np.errstate(divide='ignore'):
            psnr = 10 * np.log10((255.0 ** 2) / mse)
        
        psnr[np.isinf(psnr)] = float('inf') 

        return psnr.mean()


    def compute_ssim(self, ref, dist):
        return ssim(ref, dist, channel_axis=-1, data_range=255)



    # [B, C, H, W] at time t
    def calculate_lpips(self, frames1, frames2):

        frames1 = self.normalize_for_lpips(frames1)
        frames2 = self.normalize_for_lpips(frames2)

        B = frames1.shape[0]
        with torch.no_grad():
            scores = self.lpips_model(frames1, frames2)  # [B, 1, 1, 1]
            mean_score = scores.view(B).mean().item()
        return mean_score

    

    def compute_fvd(self,videos1, videos2, method='styleganv'):


        if method == 'styleganv':
            from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
        elif method == 'videogpt':
            from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
            from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

        print("calculate_fvd...")

        # videos [batch_size, timestamps, channel, h, w]
        
        assert videos1.shape == videos2.shape

        i3d = load_i3d_pretrained(device=self.device)
        fvd_results = []

        # support grayscale input, if grayscale -> channel*3
        # BTCHW -> BCTHW
        # videos -> [batch_size, channel, timestamps, h, w]

        videos1 = self.normalize_for_fvd(videos1)
        videos2 = self.normalize_for_fvd(videos2)

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos [batch_size, channel, timestamps, h, w]
        # get FVD features
        feats1 = get_fvd_feats(videos1, i3d=i3d, device=self.device)
        feats2 = get_fvd_feats(videos2, i3d=i3d, device=self.device)

        # calculate FVD
        fvd_result = frechet_distance(feats1, feats2)
        
        return fvd_result

    


    def evaluate(self, videos1, videos2, metrics=["psnr", "ssim", "lpips"], average=True):

        """
        Calculate metrics over time for batched videos.

        Args:
            videos1, videos2: torch.Tensor of shape [B, T, C, H, W], values in [0, 255]

        """
        assert videos1.shape == videos2.shape
        B, T, C, H, W = videos1.shape

        videos1 = videos1.to(self.device)
        videos2 = videos2.to(self.device)

        results = {m: [] for m in metrics}

        if "fvd" in metrics:
            fvd_val = self.compute_fvd(videos1, videos2, method='styleganv')
            results["fvd"].append(fvd_val)

        for t in range(T):
            frame1_np = videos1[:, t].cpu().numpy()
            frame2_np = videos2[:, t].cpu().numpy()

            if "psnr" in metrics:
                psnr_val = self.compute_psnr(frame1_np, frame2_np)
                results["psnr"].append(psnr_val)

            if "lpips" in metrics:
                lpips_val = self.calculate_lpips(videos1[:, t], videos2[:, t])
                results["lpips"].append(lpips_val)

            if "ssim" in metrics:
                ssim_vals = []
                for b in range(B):
                    ssim_val = self.compute_ssim(frame1_np[b].transpose(1, 2, 0), frame2_np[b].transpose(1, 2, 0)) # C,H,W -> H,W,C
                    ssim_vals.append(ssim_val)
                results["ssim"].append(np.mean(ssim_vals))
        
        if average:
            return {k: sum(v)/len(v) for k, v in results.items()}
        return results

    
    def normalize_for_lpips(self, x):
        return ((x.float() / 255.0) * 2 - 1).to(self.device)

    def normalize_for_fvd(self, video_tensor):
        video_tensor = video_tensor.float()/ 255.0
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        mean = torch.tensor([0.5, 0.5, 0.5], device=video_tensor.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=video_tensor.device).view(1, 3, 1, 1, 1)
        return (video_tensor - mean) / std