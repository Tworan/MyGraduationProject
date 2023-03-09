import torch 
import torch.nn as nn 
import sys
sys.path.append('/home/photon/MyGraduationProject')
import math
from models.AVfuse import AVfuse, pre_v, VideoNet
from VideoFeatureExtractor.VideoModel import VideoModel

class Encoder(nn.Module):
    def __init__(self, out_channels, kernel_size, act='relu'):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size//2,
            padding=0
        )
        if act.lower() == 'relu':
            self.act = nn.ReLU()
        elif act.lower() == 'prelu':
            self.act = nn.PReLU()
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU()
        else:
            raise NameError
        
    def forward(self, x):
        x = self.conv1d(x)
        x = self.act(x)
        return x

class Segementation(nn.Module):
    def __init__(self, K):
        super(Segementation, self).__init__()
        self.K = K
        self.P = self.K // 2
    
    def forward(self, x):
        """
            K: 语音块长度
            P: 重叠长度
            input: [B, N, L]
            output: [B, N, K, S]
        """
        B, D, L = x.shape
        x, gap = self._padding(x)
        left_part = x[:, :, :-self.P].contiguous().view(B, D, -1, self.K)
        right_part=  x[:, :, self.P:].contiguous().view(B, D, -1, self.K)
        x = torch.cat([left_part, right_part], dim=3).view(B, D, -1, self.K)
        # x: [B, N, S, K]
        x = x.transpose(2, 3)
        # x: [B, N, K, S]
        return x.contiguous(), gap 

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

class Locally_Recurrent(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=1, bidirectional=True):
        super(Locally_Recurrent, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.layerNorm = nn.LayerNorm(normalized_shape=in_channels)
        # 线形层
        self.linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels,
            in_channels
        )
    
    def forward(self, x):
        """
        input: [B, N, K, S]
        output: [B, N, K, S]
        """
        B, N, K, S = x.shape
        # x: [B, N, K, S]
        x = x.permute(0, 3, 2, 1).contiguous()
        # as follow: [batch, time, features]
        # x: [B, S, K, N]
        x = x.view(B*S, K, N)
        # x: [B*S, K, N]
        residual = x 
        x, _ = self.LSTM(x)
        # x: [B*S, K, H]
        x = self.linear(x)
        # squeeze to target size
        # x: [B*S, K, N]
        x = x + residual
        x = self.layerNorm(x)
        # x: [B*S, K, N]
        x = x.view(B, S, K, N).permute(0, 3, 2, 1).contiguous()
        # X: [B, N, K, S]
        return x 

class Positional_Encoding(nn.Module):
    """
        Positional Encoding
    """
    def __init__(self, d_model, max_len=32000 * 4):
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
        
class Globally_Attentive(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(Globally_Attentive, self).__init__()
        self.LayerNorm1 = nn.LayerNorm(normalized_shape=in_channels)
        self.Positional_Encoding = Positional_Encoding(d_model=in_channels, max_len=32000)
        self.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True 
        )
        self.LayerNorm2 = nn.LayerNorm(normalized_shape=in_channels)
        self.Dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [B, N, K, S]
        B, N, K, S = x.shape
        residual_1 = x 
        x = x.permute(0, 2, 3, 1)
        # x: [B, K, S, N]
        x = x.contiguous().view(B*K, S, N)
        x = self.LayerNorm1(x) + self.Positional_Encoding(x)
        residual_2 = x
        # residual_1: [B, N, K, S]
        # residual_2: [B*K , S, N]
        x = self.MultiheadAttention(x, x, x, attn_mask=None, key_padding_mask=None)[0]
        # x: [B*K, S, N]
        x = self.Dropout(x) + residual_2
        x = self.LayerNorm2(x)
        x = x.view(B, K, S, N)
        # x: [B, K, S, N]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x + residual_1
        return x 
    
class Sandglasset_Block(nn.Module):
    """
    <Local RNN> -> <LayrNorm> -> <Downsample> -> <Globally Attentive>
    -> <Upsample>
    """
    def __init__(self, in_channels, hidden_channels=128, num_layers=1, 
                 bidirectional=True, num_heads=8, kernel_size=1, stride=1, using_convT_to_upsample=True):
        super(Sandglasset_Block, self).__init__()

        # Local RNN
        self.Locally_Recurrent = Locally_Recurrent(in_channels=in_channels,
                                                   hidden_channels=hidden_channels,
                                                   num_layers=num_layers,
                                                   bidirectional=bidirectional)
        # LayrNorm
        self.LayerNorm = nn.LayerNorm(normalized_shape=in_channels)
        
        # Downsample
        # Use depth-wise convolution
        self.Downsample = nn.Conv1d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=0,
                                    groups=in_channels)
        # Globally Attentive
        self.Globally_Attentive = Globally_Attentive(in_channels=in_channels,
                                                     num_heads=num_heads)
        # Upsample
        if using_convT_to_upsample:
            self.Upsample = nn.ConvTranspose1d(in_channels=in_channels,
                                               out_channels=in_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=0,)
        else:
            self.Upsample = nn.Sequential(nn.Upsample(scale_factor=kernel_size, mode='linear', align_corners=True),
                                          nn.Conv1d(in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding='same',
                                                    groups=in_channels))
    
    def forward(self, x):
        """
        input: [B, N, K, S]
        output: [B, N, K, S]
        """
        B, N, K, S = x.shape
        residual = x
        x = self.Locally_Recurrent(x)
        # x: [B, N, K, S]
        x = x.permute(0, 3, 2, 1).contiguous()
        # x: [B, S, K, N]
        x = self.LayerNorm(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        # x: [B, K, N, S]
        # do downsample
        x = x.view(B*K, N, S)
        # x: [B*K, N, S]
        # print('down:', x.shape)
        x = self.Downsample(x)
        # x: [B*K, N, S/scale_factor]
        x = x.view(B, K, N, -1)
        # x: [B, K, N, S/scale_factor]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x: [B, N, K, S/scale_factor]
        x = self.Globally_Attentive(x)
        # do upsample
        x = x.permute(0, 2, 1, 3).contiguous()
        # x: [B, K, N, S/scale_factor]
        x = x.view(B*K, N, -1)
        # print('cur:', x.shape)
        x = self.Upsample(x)
        # print('up:', x.shape)
        x = x.view(B, K, N, -1).permute(0, 2, 1, 3).contiguous()
        # x: [B, N, K, S]
        return x 
    
class Separation(nn.Module):
    def __init__(self, in_channels, out_channels, length, video_inchannels, hidden_channels=128,
                 num_layers=1, bidirectional=True, num_heads=8, depth=6, speakers=2, using_convT_to_upsample=True):
        super(Separation, self).__init__()
        self.LayerNorm = nn.LayerNorm(normalized_shape=in_channels)
        # features expension
        self.Linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        
        self.Segementation = Segementation(K=length)

        self.depth = depth

        self.VidoNet = VideoNet(in_channels=video_inchannels)
        kernel_size = []
        stride = []
        # 以4为缩放尺度 每层尺度*4
        for i in range(self.depth):
            kernel_size.append(4**i)
            stride.append(4**i)
        for i in reversed(range(self.depth)):
            kernel_size.append(4**i)
            stride.append(4**i)
        
        self.Sandglasset_Blocks = nn.ModuleList([])

        for i in range(2*self.depth):
            self.Sandglasset_Blocks.append(
                Sandglasset_Block(in_channels=out_channels,
                                  hidden_channels=hidden_channels,
                                  num_layers=num_layers,
                                  bidirectional=bidirectional,
                                  num_heads=num_heads,
                                  kernel_size=kernel_size[i],
                                  stride=stride[i],
                                  using_convT_to_upsample=using_convT_to_upsample)
            )
        
        self.AVfuse_Segmentation = [Segementation(K=length) for i in range(self.depth*2)]

        self.AVfuse_net = AVfuse(audio_in_channels=out_channels,
                                 video_in_channels=video_inchannels,
                                 kernel_size=5,
                                 stride=1,
                                 groups=1,
                                 using_residual=False)

        
        self.PReLU = nn.PReLU()

        self.spk = speakers 

        self.Conv2d = nn.Conv2d(in_channels=out_channels,
                                out_channels=speakers*out_channels,
                                kernel_size=1,
                                groups=out_channels)

        self.output = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                              out_channels=out_channels,
                                              kernel_size=1),
                                    nn.Tanh())
        
        self.output_gate = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=1,
                                                   groups=out_channels),
                                         nn.Sigmoid())

        self.bottleneck = nn.Conv1d(in_channels=out_channels,
                                    out_channels=in_channels,
                                    kernel_size=1,
                                    bias=False)
        
        self.ReLU = nn.ReLU()

    def forward(self, x, v):
        """
        input: [B, N, L]
        output: [B, N, L]
        """
        x = x.permute(0, 2, 1).contiguous()
        # x: [B, L, N]
        x = self.LayerNorm(x)
        x = self.Linear(x).permute(0, 2, 1).contiguous()
        # x: [B, N, L]
        x, _gap = self.Segementation(x)
        # 残差连接
        self.residual = []
        gap = _gap
        for i in range(2*self.depth):
            x = self.Sandglasset_Blocks[i](x)
            # v = self.VidoNet(v)
            # reshape
            # x: [B, N, K, S]
            # AVfuse
            residual = x
            x = self._overlap_add(x, gap)
            x, v = self.AVfuse_net(x, v)
            x, gap = self.AVfuse_Segmentation[i](x)   
            
            if i < self.depth:
                self.residual.append(residual)
            else:
                # sum fuse
                x = x + self.residual[2*self.depth - 1 - i]
                # cat fuse
                # x = torch.cat([x, self.residual[2*self.depth - 1 - i]], dim=-1)
        x = self.PReLU(x)

        x = self.Conv2d(x)
        # x: [B, speakers*out_channels, K, S]
        B, _, K, S = x.shape

        # do audio reconstruct
        x = x.view(B * self.spk, -1, K, S)
        x = self._overlap_add(x, _gap)
        # x: [B * spk, N, L]
        x = self.output(x) * self.output_gate(x)
        # x: [B * spk, output_channels, L]
        x = self.bottleneck(x)
        # x: [B * spk, N, L]
        _, N, L = x.shape
        x = x.view(B, self.spk, N, L)
        
        x = self.ReLU(x)

        x = x.permute(1, 0, 2, 3).contiguous()
        # x: [spk, B, N, L]
        return x


    def _overlap_add(self, x, gap):
        """
        input: [B, N, K, S]
        output: [B, N, L]
        """
        B, N, K, S = x.shape 
        P = K // 2
        
        x = x.permute(0, 1, 3, 2).contiguous()
        # x: [B, N, S, K]
        x = x.view(B, N, -1, K * 2)
        left = x[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        right = x[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        audio = left + right 
        # audio: [B, N, L]
        return audio[:, :, :-gap]

class Decoder(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(Decoder, self).__init__()

        self.ConvT1d = nn.ConvTranspose1d(in_channels=in_channels,
                                          out_channels=1,
                                          kernel_size=kernel_size,
                                          stride=kernel_size//2,
                                          padding=0
                                          )
    
    def forward(self, x):
        """
        input: [spk, B, N, L]
        output: [spk, B, 1, L]
        """
        x = self.ConvT1d(x)
        return x 

class AVfusedSandglasset(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, length, video_model, video_inchannels=512, hidden_channels=128,
                 num_layers=1, bidirectional=True, num_heads=8, depth=3, speakers=2, using_convT_to_upsample=True):
        super(AVfusedSandglasset, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = self.kernel_size // 2
        self.length = length
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.depth = depth
        self.using_convT_to_upsample = using_convT_to_upsample
        self.speakers = speakers

        self.pre_v = pre_v(videomodel=video_model)

        # self.video_model = video_model

        self.Encoder = Encoder(out_channels=in_channels, kernel_size=kernel_size)

        self.Separation = Separation(in_channels=in_channels,
                                     out_channels=out_channels,
                                     length=length,
                                     hidden_channels=hidden_channels,
                                     video_inchannels=video_inchannels,
                                     num_layers=num_layers,
                                     bidirectional=bidirectional,
                                     num_heads=num_heads,
                                     depth=depth,
                                     speakers=1,
                                     using_convT_to_upsample=using_convT_to_upsample)
        
        self.Decoder = Decoder(in_channels=in_channels, kernel_size=kernel_size)

    def forward(self, x, v):
        """
        input: [B, 1, L]
        output: [B, spk, L]
        """
        # x, 
        # print(x.shape)
        x, gap = self._padding(x)
        e = self.Encoder(x)
        # v
        v = self.pre_v(v)
        
        m = self.Separation(e, v)
        out = m[0] * e 
        # outs = [m[i] * e for i in range(self.speakers)]
        audios = self.Decoder(out)[:, :, self.stride: -(gap+self.stride)]
        # audios: [[B, 1, L]]
        return audios

    def _padding(self, x):
        """
        describe: 填充至P的整数倍
        input:  [B, 1, L]
        output: [B, 1, L]
        """
        B, _, L = x.shape
        gap = self.kernel_size - L % (self.stride)
        #     200 - 31999 % 100
        pad = torch.zeros([B, 1, gap]).to(x.device, dtype=torch.float)
        x = torch.cat([x, pad], dim=2)
        _pad = torch.zeros(size=(B, 1, self.stride)).to(x.device, dtype=torch.float)
        x = torch.cat([_pad, x, _pad], dim=2)
        return x, gap 

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package, conf):
        model = cls(in_channels=package['in_channels'], out_channels=package['out_channels'],
                    kernel_size=package['kernel_size'], length=package['length'],
                    hidden_channels=package['hidden_channels'], num_layers=package['num_layers'],
                    bidirectional=package['bidirectional'], num_heads=package['num_heads'],
                    depth=package['depth'], speakers=package['speakers'], video_model=conf['model']['sandglasset']['video_model'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'in_channels': model.in_channels, 'out_channels': model.out_channels,
            'kernel_size': model.kernel_size, 'length': model.length,
            'hidden_channels': model.hidden_channels, 'num_layers': model.num_layers,
            'bidirectional': model.bidirectional, 'num_heads': model.num_heads,
            'depth': model.depth, 'speakers': model.speakers,
            'mode': 'audio-visual',
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

if __name__ == '__main__':
    input_audio = torch.randn(size=(1, 1, 32000)).to('cpu')
    input_video = torch.randn(size=(1, 2, 100, 96, 96)).to('cpu')
    # []
    model = VideoModel()
    # video_features = model(input_video)
    # print(video_features.shape)
    state_dict = torch.load('frcnn_128_512.backbone.pth.tar')['model_state_dict']
    # print(state_dict)
    model.load_state_dict(state_dict)
    # model(input_video)
    print('model load successfully')
    
    model = AVfusedSandglasset(in_channels=256,
                        out_channels=64,
                        kernel_size=38,
                        length=256,
                        hidden_channels=256,
                        num_layers=1,
                        video_inchannels=512,
                        bidirectional=True,
                        num_heads=8,
                        depth=3,
                        speakers=2,
                        video_model='frcnn_128_512.backbone.pth.tar',
                        using_convT_to_upsample=False).cpu()
    y = model(input_audio, input_video)
    print('pass')
