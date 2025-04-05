import cv2

from torch.utils.data import DataLoader

# class video_dataset(torch.utils.data.Dataset):
#     def __init__(self, video_path):
#         self.video_files = []
#         for video_file in os.listdir(video_path):
#             if video_file.endswith('.mp4'):
#                 self.video_files.append(os.path.join(video_path, video_file))
#         self.video_files.sort()
#     def resize_frame(self, frame):
#         return cv2.resize(frame, (512, 512))
#     def load_video(self, video_file):
#         cap = cv2.VideoCapture(video_file)
#         video = []
#         while True:
#             ret, frame = cap.read()
#             frame = self.resize_frame(frame)
#             if not ret:
#                 break
#             video.append(torch.from_numpy(frame.transpose(2,0,1).astype('float32') / 255))

#         return torch.stack(video,dim=1) # [B, C, T, H, W]
    
#     def __len__(self):
#         return len(self.video_files)
    
#     def __getitem__(self, idx):
#         video_file = self.video_files[idx]
#         video = self.load_video(video_file)
#         video_name = video_file.split('/')[-1]
#         return video, video_name




def video_to_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames