import torch 
import torch.nn as nn
import torch.nn.functional as F
from VideoFeatureExtractor.VideoModel import VideoModel
import math

class Positional_Encoding(nn.Module):
    """
        Positional Encoding
    """
    def __init__(self, d_model, max_len=32000):
        """
        d_model: Feature
        max_len: max lens of the seqs
        """
        super(Positional_Encoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # position: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
           input: [N, T, D]
        """
        length = x.size(1)
        return self.pe[:, :length]
    

class VideoSegementation(nn.Module):
    def __init__(self, K):
        super(VideoSegementation, self).__init__()
        self.K = K
        self.P = K // 2
    
    def forward(self, v, a):
        v = F.interpolate(v, size=a.shape[-1], mode='nearest')
        B, D, L = v.shape
        v, gap = self._padding(v)
        left_part = v[:, :, :-self.P].contiguous().view(B, D, -1, self.K)
        right_part=  v[:, :, self.P:].contiguous().view(B, D, -1, self.K)
        v = torch.cat([left_part, right_part], dim=3).view(B, D, -1, self.K)
        # x: [B, N, S, K]
        v = v.transpose(2, 3)
        # x: [B, N, K, S]
        return v.contiguous(), gap 
    
    def _padding(self, x):
        """
        describe: 填充至P的整数倍
        input:  [B, N, L]
        output: [B, N, L]
        """
        B, N, L = x.shape
        gap = self.K - L % self.P
        #     200 - 31999 % 100
        pad = torch.zeros([B, N, gap]).to(x.device, dtype=torch.float)
        x = torch.cat([x, pad], dim=2)
        _pad = torch.zeros(size=(B, N, self.P)).to(x.device, dtype=torch.float)
        x = torch.cat([_pad, x, _pad], dim=2)
        return x, gap 

class VideoRencoder(nn.Module):
    def __init__(self, out_channels, video_embeded_size=512):
        """
        descripition: preprocess before the avfuse. 
        input: [B, n_src, T, H, W]
        output: [B, n_channels, T]
        """
        super(VideoRencoder, self).__init__()
        self.out_channels = out_channels
        self.video_embeded_size = video_embeded_size
        self.pre_conv = torch.nn.ConvTranspose1d(in_channels=video_embeded_size,
                                                out_channels=out_channels,
                                                kernel_size=8,
                                                stride=video_embeded_size//out_channels,
                                                padding=4)


    def forward(self, v):
        # print(v.shape)
        B, N, T = v.shape
        v = self.pre_conv(v)
        return v

class CrossModalAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(CrossModalAttention, self).__init__()
        self.LayerNorm1_s = nn.LayerNorm(normalized_shape=in_channels)
        self.LayerNorm1_v = nn.LayerNorm(normalized_shape=in_channels)

        self.Positional_Encoding = Positional_Encoding(d_model=in_channels, max_len=32000)
        self.MultiheadCrossModalAttention_s2v = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True 
        )
        self.MultiheadCrossModalAttention_v2s = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True 
        )

        self.GlobalAttention_s2s = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True 
        )

        self.GlobalAttention_v2v = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True 
        )
        self.LayerNorm2_s = nn.LayerNorm(normalized_shape=in_channels)
        self.LayerNorm2_v = nn.LayerNorm(normalized_shape=in_channels)
        self.Dropout_s = nn.Dropout(p=0.1)
        self.Dropout_v = nn.Dropout(p=0.1)
        # self._lambda_s_s = torch.Tensor(1.)
        # self._lambda_s_v = torch.Tensor(1.)
        # self._lambda_v_v = torch.Tensor(1.)
        # self._lambda_v_s = torch.Tensor(1.)
        self.gate_v = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=1,),
            nn.Sigmoid()
        )

        self.gate_a = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=1,),
            nn.Sigmoid()
        )

    def forward(self, s, v):
        """
        s: [B, N, K, S]
        v: [B, N, K, S]
        """
        B, N, K, S = s.shape 
        residual_s1 = s
        residual_v1 = v
        s = s.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        v = v.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)

        # [B*K, S, N]
        s = self.LayerNorm1_s(s)
        gate_a = self.gate_a(s.transpose(1, 2)).transpose(1, 2)
        s = s + self.Positional_Encoding(s)

        v = self.LayerNorm1_v(v)
        gate_v = self.gate_v(v.transpose(1, 2)).transpose(1, 2)        
        v = v + self.Positional_Encoding(v)

        residual_s2 = s
        residual_v2 = v
        # audio 通过query video得每一帧 得到对应的
        v2s = self.MultiheadCrossModalAttention_v2s(v, s, s, attn_mask=None, key_padding_mask=None)[0]
        s2v = self.MultiheadCrossModalAttention_s2v(s, v, v, attn_mask=None, key_padding_mask=None)[0]
        s2s = self.GlobalAttention_s2s(s, s, s, attn_mask=None, key_padding_mask=None)[0]
        v2v = self.GlobalAttention_v2v(v, v, v, attn_mask=None, key_padding_mask=None)[0]

        s = s2s + gate_a * v2s
        v = v2v + gate_v * s2v 

        # s = residual_s2 + self.Dropout_s(s)
        # v = residual_v2 + self.Dropout_v(v) 

        s = residual_s2 + s
        v = residual_v2 + v

        s = self.LayerNorm2_s(s)
        v = self.LayerNorm2_v(v)

        s = s.view(B, K, S, N)
        v = v.view(B, K, S, N)

        s = s.permute(0, 3, 1, 2).contiguous()
        v = v.permute(0, 3, 1, 2).contiguous()

        s = s + residual_s1
        v = v + residual_v1

        return s, v
    

class VideoNet(nn.Module):
    def __init__(self, in_channels, hidden_size=128, num_layers=1, bidirectional=True, _type='LSTM'):
        super(VideoNet, self).__init__()
        self._type = _type
        if self._type == 'LSTM':
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
        elif self._type == 'MHA':
            self.MHA = nn.MultiheadAttention(
                embed_dim=in_channels,
                num_heads=8,
                dropout=0.1
            )
            self.layernorm = nn.LayerNorm(normalized_shape=in_channels)
        
    def forward(self, x):
        """
        input: [B, N, T]
        """
        x = x.permute(0, 2, 1).contiguous()
        # x: [B, T, N]
        residual = x
        if self._type == 'LSTM':
            x, _ = self.LSTM(x)
            x = self.linear(x)
            x += residual
            x = self.layernorm(x)
        elif self._type == 'MHA':
            x, _ = self.MHA(x)
            x += residual
            x = self.layernorm(x)
        x = x.permute(0, 2, 1).contiguous()
        return x

class pre_v(nn.Module):
    def __init__(self, videomodel, video_embeded_size=512, kernel_size=5, n_src=1):
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
        # self.spks_fuse = nn.Sequential(
        #     nn.Conv2d(in_channels=video_embeded_size,
        #               out_channels=video_embeded_size,
        #               kernel_size=(n_src, kernel_size),
        #               stride=1,
        #               padding=[0, kernel_size//2],
        #               groups=video_embeded_size),
        #     # nn.ReLU()
        # )

    def forward(self, v):
        B, n_src, T, H, W = v.shape
        v = v.contiguous()
        v = v.view(B*n_src, 1, T, H, W)
        with torch.no_grad():
            v = self.videomodel(v)
        # _, Nnew, Tnew = v.shape
        # v = v.view(B, n_src, Nnew, Tnew)
        # v = v.permute(0, 2, 1, 3).contiguous()
        # # v = self.spks_fuse(v)
        # # v: [B, Nnew, 1, Tnew]
        # v = v.permute(0, 2, 1, 3).contiguous().squeeze(1)
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

        self.video_norm = nn.LayerNorm(video_in_channels)

            
            # nn.ReLU()
        self.video_conv = nn.Conv1d(in_channels=video_in_channels+audio_in_channels, 
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

        self.audio_gate = nn.Sequential(
            nn.Conv1d(in_channels=audio_in_channels,
                      out_channels=audio_in_channels,
                      kernel_size=1,),
            nn.Tanh()
        )

        self.video_gate = nn.Sequential(
            nn.Conv1d(in_channels=video_in_channels,
                      out_channels=video_in_channels,
                      kernel_size=1,),
            nn.Tanh()
        )
    
    def forward(self, a, v):
        """
        input:
            a: [B, N, L]
            v: [B, N, L]
        """
        # print(a.shape, v.shape)
        residual_a = a
        residual_v = v

        # v = self.video_upsample(v)
        # v = v.permute(0, 2, 1).contiguous()
        # v = self.video_sub_norm(v)
        # v = v.permute(0, 2, 1).contiguous()

        # a = self.audio_downsample(a)
        # a = a.permute(0, 2, 1).contiguous()
        # a = self.audio_sub_norm(a)
        # a = a.permute(0, 2, 1).contiguous()

        v_gate = self.video_gate(v)
        a_gate = self.audio_gate(a)

        sa = F.interpolate(v, size=residual_a.shape[-1], mode='nearest')
        sv = F.interpolate(a, size=residual_v.shape[-1], mode='nearest')
        # print(a.shape, sa.shape, v.shape, sv.shape)
        # print(residual_v.shape, residual_a.shape)
        a = torch.cat([a, sa], dim=1)
        v = torch.cat([v, sv], dim=1)

        a = self.audio_conv(a)
        v = self.video_conv(v)

        a = a.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()

        a = self.audio_norm(a)
        v = self.video_norm(v)

        a = a.permute(0, 2, 1).contiguous() * a_gate
        v = v.permute(0, 2, 1).contiguous() * v_gate

        if self.using_residual:
            a = a + residual_a
            v = v + residual_v
        return a, v