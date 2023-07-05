import torch 
import decord as de
import torch.utils.data as data
ctx = de.cpu(0)
de.bridge.set_bridge("torch")

import skvideo.io
import skvideo.datasets


class BigBuckBunny(data.Dataset):
    def __init__(
        self,
        frame_interval=1
    ):
        self.frame_interval = frame_interval
        self.video = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
    
    def __len__(self):
        return self.video.shape[0] // self.frame_interval
    
    def __getitem__(self, idx):
        idx = idx * self.frame_interval
        frame = self.video[idx] / 255
        idx = idx / (len(self.video) / self.frame_interval) # normalize idx
        return torch.tensor(idx), torch.Tensor(frame).float().permute(2, 0, 1)
    
class VideoDataset(data.Dataset):
    def __init__(
        self, 
        video_path,
        frame_interval=1,
    ):
        self.frame_interval = frame_interval
        self.video = de.VideoReader(video_path, ctx=ctx)
    
    def __len__(self):
        return len(self.video) // self.frame_interval
    
    def __getitem__(self, idx):
        idx = idx * self.frame_interval
        frame = self.video[idx].permute(2, 0, 1)
        idx = idx / (len(self.video) / self.frame_interval) # normalize idx
        return torch.tensor(idx), frame