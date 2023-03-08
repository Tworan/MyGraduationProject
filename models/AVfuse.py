import torch 
import torch.nn as nn
import torch.nn.functional as F
from VideoFeatureExtractor.VideoModel import VideoModel

class VideoNet(nn.Module):
    def __init__(self, in_channels, hidden_size=256, num_layers=1, bidirectional=True):
        super(VideoNet, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional)
        self.layernorm = nn.LayerNorm(normalized_shape=in_channels)
        self.linear = nn.Linear(
            hidden_size*2 if bidirectional else hidden_size,
            in_channels
        )
    
    def forward(self, x):
        """
        input: [B, N, T]
        """
        x = x.permute(0, 2, 1).contiguous()
        # x: [B, T, N]
        residual = x
        x, _ = self.LSTM(x)
        x = self.linear(x)
        x += residual
        x = self.layernorm(x)
        x = x.permute(0, 2, 1).contiguous()
        return x

class pre_v(nn.Module):
    def __init__(self, videomodel, video_embeded_size=512, kernel_size=5, n_src=2):
        """
        descripition: preprocess before the avfuse. 
        input: [B, n_src, T, H, W]
        output: [B, n_channels, T]
        """
        super(pre_v, self).__init__()
        self.videomodel = VideoModel()
        # video_features = model(input_video)
        # print(video_features.shape)
        self.videomodel.load_state_dict(torch.load(videomodel)['model_state_dict'])
        self.spks_fuse = nn.Sequential(
            nn.Conv2d(in_channels=video_embeded_size,
                      out_channels=video_embeded_size,
                      kernel_size=(n_src, kernel_size),
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
        self.audio_conv = nn.Conv1d(in_channels=audio_in_channels*2, 
                                    out_channels=audio_in_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding='same',
                                    groups=groups)
        self.audio_norm = nn.LayerNorm(audio_in_channels)

        self.video_norm = nn.LayerNorm(video_in_channels)

            
            # nn.ReLU()
        self.video_conv = nn.Conv1d(in_channels=video_in_channels*2, 
                                    out_channels=video_in_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding='same',
                                    groups=groups)
        
        self.video_upsample = nn.Sequential(
            nn.ConvTranspose1d(in_channels=video_in_channels,
                               out_channels=video_in_channels // 2,
                               kernel_size=4,
                               stride=2,
                               padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=video_in_channels // 2,
                               out_channels=video_in_channels // 4 ,
                               kernel_size=4,
                               stride=2,
                               padding=0),
        )
        self.video_sub_norm = nn.LayerNorm(video_in_channels // 4)

        self.audio_downsample = nn.Sequential(
            nn.Conv1d(in_channels=audio_in_channels,
                      out_channels=audio_in_channels * 2,
                      kernel_size=4,
                      stride=2,
                      padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=audio_in_channels * 2,
                      out_channels=audio_in_channels * 4,
                      kernel_size=4,
                      stride=2,
                      padding=0),  
        )
        self.audio_sub_norm = nn.LayerNorm(audio_in_channels * 4)
            # nn.ReLU()

        self.using_residual = using_residual
    
    def forward(self, a, v):
        """
        input:
            a: [B, N, L]
            v: [B, N, L]
        """
        # print(a.shape, v.shape)
        residual_a = a
        residual_v = v
        v = self.video_upsample(v)
        v = v.permute(0, 2, 1).contiguous()
        v = self.video_sub_norm(v)
        v = v.permute(0, 2, 1).contiguous()

        a = self.audio_downsample(a)
        a = a.permute(0, 2, 1).contiguous()
        a = self.audio_sub_norm(a)
        a = a.permute(0, 2, 1).contiguous()


        sa = F.interpolate(v, size=residual_a.shape[-1], mode='nearest')
        sv = F.interpolate(a, size=residual_v.shape[-1], mode='nearest')
        # print(a.shape, sa.shape, v.shape, sv.shape)
        # print(residual_v.shape, residual_a.shape)
        a = torch.cat([residual_a, sa], dim=1)
        v = torch.cat([residual_v, sv], dim=1)

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