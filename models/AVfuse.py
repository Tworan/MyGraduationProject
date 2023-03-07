import torch 
import torch.nn as nn
import torch.nn.functional as F
from VideoFeatureExtractor.VideoModel import VideoModel

class pre_v(nn.Module):
    def __init__(self, videomodel, video_embeded_size=512, kernel_size=5):
        """
        descripition: preprocess before the avfuse. 
        input: [B, n_src, T, H, W]
        output: [B, T, n_channels]
        """
        super(pre_v, self).__init__()
        self.videomodel = VideoModel()
        # video_features = model(input_video)
        # print(video_features.shape)
        self.videomodel.load_state_dict(torch.load(videomodel)['model_state_dict'])
        self.spks_fuse = nn.Sequential(
            nn.Conv2d(in_channels=video_embeded_size,
                      out_channels=video_embeded_size,
                      kernel_size=(2, kernel_size),
                      stride=1,
                      padding=[0, kernel_size//2],
                      groups=video_embeded_size),
            # nn.ReLU()
        )

    def forward(self, v):
        B, n_src, T, H, W = v.shape
        v = v.contiguous()
        v = v.view(B*n_src, 1, T, H, W)
        with torch.no_grad():
            v = self.videomodel(v)
        _, Nnew, Tnew = v.shape
        v = v.view(B, n_src, Nnew, Tnew)
        v = v.permute(0, 2, 1, 3).contiguous()
        v = self.spks_fuse(v)
        # v: [B, Nnew, 1, Tnew]
        v = v.permute(0, 2, 1, 3).contiguous().squeeze(1)
        # v: [B, Nnew, Tnew]
        return v



class AVfuse(nn.Module):
    def __init__(self, audio_in_channels, video_in_channels, kernel_size=5, stride=1, groups=1, using_residual=False):
        """
        接受audio和vision数据直接融合
        """
        super(AVfuse, self).__init__()
        self.audio_conv = nn.Conv1d(in_channels=audio_in_channels+video_in_channels, 
                                    out_channels=audio_in_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding='same',
                                    groups=groups)
        self.audio_norm = nn.LayerNorm(audio_in_channels)
            
            # nn.ReLU()
        self.video_conv = nn.Conv1d(in_channels=audio_in_channels+video_in_channels, 
                                    out_channels=video_in_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding='same',
                                    groups=groups)
        self.video_norm = nn.LayerNorm(video_in_channels)
            # nn.ReLU()

        self.using_residual = using_residual
    
    def forward(self, a, v):
        """
        input:
            a: [B, N, L]
            v: [B, N, L]
        """
        residual_a = a
        residual_v = v
        sa = F.interpolate(v, size=a.shape[-1], mode='nearest')
        sv = F.interpolate(a, size=v.shape[-1], mode='nearest')
        a = torch.cat([a, sa], dim=1)
        v = torch.cat([v, sv], dim=1)

        a = self.audio_conv(a)
        v = self.video_conv(v)

        a = a.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()

        a = self.audio_norm(a)
        v = self.video_norm(v)

        a = a.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()

        if self.using_residual:
            a = a + residual_a
            v = v + residual_v
        return a, v