import torch.nn as nn
import torch
import sys
sys.path.append('../')
from models.ResNet import ResNet, BasicBlock
from torch.nn.modules.batchnorm import _BatchNorm


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


class VideoModel(nn.Module):
    def __init__(self, 
                hidden_dim=256, 
                relu_type="prelu", 
                width_mult=1.0,
                pretrain=None):
        super(VideoModel, self).__init__()
    
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
    

        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout) if relu_type == "prelu" else nn.ReLU()
        )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.pretrain = pretrain
        if pretrain:
            self.init_from(pretrain)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        return x.transpose(1, 2)

    def init_from(self, path):
        pretrained_dict = torch.load(path, map_location="cpu")["model_state_dict"]
        update_frcnn_parameter(self, pretrained_dict)

    def train(self, mode=True):
        super().train(mode)
        if mode:    # freeze BN stats
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def check_parameters(net):
    """
    Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


def update_frcnn_parameter(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if "tcn" in k:
            pass
        else:
            update_dict[k] = v
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        p.requires_grad = False
    return model


if __name__ == "__main__":
    frames = torch.randn(1, 1, 10, 96, 96)
    model = VideoModel()
    model.load_state_dict(torch.load('../frcnn_128_512.backbone.pth.tar')['model_state_dict'])
    print('model load successfully')
    out = model(frames)
    print(out.shape)
