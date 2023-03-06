import torch 
import torch.nn as nn
import torch.nn.functional as F

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