# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from .gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=1, input_size=(3, 512, 512),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//16, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//16),
            act_layer(act),
            nn.Conv2d(out_dim//16, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self,in_channels, out_channels=320, k=9, conv='mr', act='gelu', norm='batch', bias=True, n_blocks=16, use_dilation=True, epsilon=0.2, use_stochastic=False, drop_path=0.3):
        super(DeepGCN, self).__init__()
        
        self.n_blocks = n_blocks
        self.stem = Stem(in_dim=in_channels, out_dim=out_channels, act=act)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, out_channels, 32, 32))

        if use_dilation:
            self.backbone = Seq(*[Seq(Grapher(out_channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, use_stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(out_channels, out_channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(out_channels, num_knn[i], 1, conv, act, norm,
                                                bias, use_stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(out_channels, out_channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])


        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        _, _, Orig_H, Orig_W = inputs.shape
        pos_embed = F.interpolate(self.pos_embed, size=(int(Orig_H/32), int(Orig_W/32)), mode="bicubic")
        x = self.stem(inputs) + pos_embed
        # x = self.stem(inputs)
        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
        


        return x

def vig_ti_224_gelu(in_channels=1, out_channels=512, k=9, n_blocks=6, pretrained=False, weight_path='', **kwargs):
    model = DeepGCN(in_channels=in_channels, out_channels=out_channels, k=k, n_blocks=n_blocks)
    model.default_cfg = default_cfgs['gnn_patch16_224']

    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model
