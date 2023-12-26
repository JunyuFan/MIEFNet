import os
import sys
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import drop_path
import timm

from .vig import vig_ti_224_gelu
from .ffm import FeatureFusionModule


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class ReLUBN(nn.Sequential):
    def __init__(self, dim, dropout=0.3, norm_layer=nn.BatchNorm2d):
        super(ReLUBN, self).__init__(
            nn.ReLU(),
            norm_layer(dim),
            nn.Dropout(p=dropout)
        )

class ChannelAttentionModul(nn.Module): 
    def __init__(self, in_channel, r=0.5):  
        super(ChannelAttentionModul, self).__init__()

        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)), 
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_branch = self.MaxPool(x)

        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)


        avg_branch = self.AvgPool(x)

        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)


        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)


        h, w = weight.shape
        Mc = torch.reshape(weight, (h, w, 1, 1))

        x = Mc * x

        return x

class SpatialAttentionModul(nn.Module):
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        MaxPool = torch.max(x, dim=1).values  
        AvgPool = torch.mean(x, dim=1)

        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  

        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        x = Ms * x

        return x

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel) 
        self.Sam = SpatialAttentionModul(in_channel=in_channel) 

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x

class DWCT(nn.Module):
    def __init__(self,in_ch,out_ch, kernel_size=3, stride=1):
        super(DWCT, self).__init__()
        self.depth_conv_trans = nn.ConvTranspose2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv_trans(input)
        out = self.point_conv(out)
        return out

class RGFS(nn.Module):
    def __init__( self, input_channel=256, output_channel=256, ratio=1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1):
        super(RGFS, self).__init__()
        
        self.output_channel = output_channel
        self.conv1 = nn.Conv2d(input_channel, int(output_channel*ratio), kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
    def forward(self, x, dsm):
        inter_feature = self.conv1(x)
        inter_feature = inter_feature.permute(0, 2, 3, 1)

        dsm = torch.squeeze(dsm, dim=1)

        y = torch.zeros((inter_feature.size()[:3] + (self.output_channel,))).permute(0, 3, 1, 2).cuda()
        for i in range(10):
            index = torch.zeros((x.size()[0], x.size()[2], x.size()[3], 1)).cuda()
            index[dsm==i] = 1
            temp = torch.mul(inter_feature, index)
            sum_ = temp.sum(dim=1).sum(dim=1)
            _, indices = torch.sort(sum_, descending=True)
            e, _ = indices[:, :self.output_channel].sort()
            for j in range(inter_feature.size()[0]):
                y[j] += temp.permute(0, 3, 1, 2)[j, e[j], :, :]
        return y

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class FeatureRefinementHead(nn.Module):
    def __init__(self, skip_channels, feat_channels):
        super().__init__()
        self.pre_conv = Conv(skip_channels, feat_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(feat_channels, feat_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, groups=feat_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(feat_channels, feat_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(feat_channels//16, feat_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(feat_channels, feat_channels, kernel_size=1)
        self.proj = SeparableConvBN(feat_channels, feat_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.x_qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.y_qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.x_proj = nn.Linear(all_head_dim, dim)
        self.y_proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        x_qkv = F.linear(input=x, weight=self.x_qkv.weight, bias=qkv_bias)
        x_qkv = x_qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        y_qkv = F.linear(input=y, weight=self.y_qkv.weight, bias=qkv_bias)
        y_qkv = y_qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        y_q, y_k, y_v = y_qkv[0], y_qkv[1], y_qkv[2]  

        x_q = x_q * self.scale
        x_attn = (x_q @ x_k.transpose(-2, -1))
        
        y_q = y_q * self.scale
        y_attn = (y_q @ y_k.transpose(-2, -1))

        
        x_attn = self.attn_drop(x_attn)
        x_attn = x_attn.softmax(dim=-1)

        y_attn = self.attn_drop(y_attn)
        y_attn = y_attn.softmax(dim=-1)

        x = (x_attn @ y_v).transpose(1, 2).reshape(B, N, -1)
        x = self.x_proj(x)
        x = self.proj_drop(x)

        y = (y_attn @ x_v).transpose(1, 2).reshape(B, N, -1)
        y = self.y_proj(y)
        y = self.proj_drop(y)
        return x, y

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.x_norm1 = norm_layer(dim)
        self.y_norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.x_norm2 = norm_layer(dim)
        self.y_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, y):
        if self.gamma_1 is None:
            x = self.x_norm1(x)
            y = self.y_norm1(y)
            x, y = self.attn(x, y)
            x = x + self.drop_path(x)
            y = y + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.x_norm2(x)))
            y = y + self.drop_path(self.mlp(self.y_norm2(y)))

        else:
            x = self.x_norm1(x)
            y = self.y_norm1(y)
            x, y = self.attn(x, y)
            x = x + self.drop_path(self.gamma_1 * x)
            y = y + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.x_norm2(x)))
            y = y + self.drop_path(self.gamma_2 * self.mlp(self.y_norm2(y)))

        return x, y


def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.3,
                 drop_path_rate=0.3, norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False, num_patch_size=64):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_patch_size = num_patch_size


        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patch_size**2, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patch_size**2, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, y):
        B, C, H, W = x.shape    #384, 64, 64
        x = rearrange(x, 'b c h w -> b (h w) c', b=B, h=H, w=W, c=C)
        y = rearrange(y, 'b c h w -> b (h w) c', b=B, h=H, w=W, c=C)

        pos_embed = rearrange(self.pos_embed, 'b (h w) d -> b d h w', d=self.embed_dim, h=self.num_patch_size, w=self.num_patch_size)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic')
        pos_embed = rearrange(pos_embed, 'b d h w -> b (h w) d', h=H, w=W, d=self.embed_dim)
        x = x + pos_embed.type_as(x).to(x.device).clone().detach()
        y = y + pos_embed.type_as(y).to(y.device).clone().detach()

        for blk in self.blocks:
            x, y = blk(x, y)
        x = x + y
        x = self.norm(x)
        x = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W, d=self.embed_dim)
        return x

    def forward(self, x, y):
        x = self.forward_features(x, y)
        return x

class Mix(nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, x, y):
        assert x.shape == y.shape
        b, c, h, w = x.shape
        x_ = rearrange(x, 'b c h w -> b c (h w)', c=c, h=h, w=w)
        y_ = rearrange(y, 'b c h w -> b c (h w)', c=c, h=h, w=w)
        x_norm = x_ / x_.norm(dim=2, keepdim=True)
        y_norm = y_ / y_.norm(dim=2, keepdim=True)
        m = x_norm * y_norm
        m = rearrange(m, 'b c (h w) -> b c h w', c=c, h=h, w=w)

        return m

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = ConvBNReLU(in_channels, out_channels, 3, stride=1)
        self.conv_1x1 = ConvBN(out_channels, out_channels, 1, stride=1)

    def forward(self, x, fre=False):
        """Forward function."""
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten

class RelationAwareFusion(nn.Module):
    def __init__(self, channels, ext=2, r=32):
        super(RelationAwareFusion, self).__init__()
        self.r = r
        scale = (r * r) // channels
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.spatial_mlp = nn.Sequential(nn.Linear(channels * scale, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_att = ChannelAtt(channels * ext, channels)
        self.context_mlp = nn.Sequential(*[nn.Linear(channels * scale, channels), nn.ReLU(), nn.Linear(channels, channels)])
        self.context_att = ChannelAtt(channels, channels)
        self.context_head = ConvBNReLU(channels, channels, 3, stride=1)
        self.smooth = ConvBN(channels, channels, 3, stride=1)

    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        s_feat, s_att = self.spatial_att(sp_feat)   #64
        c_feat, c_att = self.context_att(co_feat)   #64
        b, c, h, w = s_att.size()
        s_att_split = s_att.view(b, self.r, c // self.r)        # b, 16, 4
        c_att_split = c_att.view(b, self.r, c // self.r)        # b, 16, 4
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))     # 16, 4 * 4, 16 = 16 * 16      
        chl_affinity = chl_affinity.view(b, -1)     # b, 256
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))
        c_feat = torch.mul(c_feat, re_c_att)
        s_feat = torch.mul(s_feat, re_s_att)

        return c_feat, s_feat



class MIEFNet(nn.Module):
    def __init__(self, dim=512, vis_channels=3, modal_channels=1, num_classes=6) -> None:
        super().__init__()

        self.dim = dim

        self.cnn_backbone = timm.create_model('swsl_resnext50_32x4d', in_chans=vis_channels+1, features_only=True, out_indices=(1, 2, 3, 4), pretrained=True)
        vis_encoder_channels = self.cnn_backbone.feature_info.channels()

        self.vis_dim1 = ConvBNReLU(vis_encoder_channels[0], dim//8, kernel_size=1)
        self.vis_dim2 = ConvBNReLU(vis_encoder_channels[1], dim//4, kernel_size=1)
        self.vis_dim3 = ConvBNReLU(vis_encoder_channels[2], dim//2, kernel_size=1)
        self.vis_dim4 = ConvBNReLU(vis_encoder_channels[3], dim, kernel_size=1)


        self.gcn_backbone = vig_ti_224_gelu(in_channels=1, out_channels=dim, k=12, n_blocks=6)

        self.informationChange = RelationAwareFusion(dim, ext=1)



        self.encoder = VisionTransformer(embed_dim=dim, depth=1, init_values=0, num_patch_size=64, num_heads=8)
        self.conv1 = ConvBNReLU(dim, dim//2, kernel_size=1)
        self.dsm_decoder1 = ConvBNReLU(dim, dim//2, kernel_size=3)

        self.ffm1 = FeatureFusionModule(dim//2, reduction=1, num_heads=8)

        self.conv2 = ConvBNReLU(dim//2, dim//4, kernel_size=1)
        self.dsm_decoder2 = ConvBNReLU(dim//2, dim//4, kernel_size=3)

        self.ffm2 = FeatureFusionModule(dim//4, reduction=1, num_heads=8)
        
        
        self.conv3 = ConvBNReLU(dim//4, dim//8, kernel_size=1)
        self.dsm_decoder3 = ConvBNReLU(dim//4, dim//8, kernel_size=3)

        self.ffm3 = FeatureFusionModule(dim//8, reduction=1, num_heads=8)

        self.segmentation_head = nn.Sequential(ConvBNReLU(dim//8, dim//16),
                                               nn.Dropout2d(p=0.3, inplace=True),
                                               Conv(dim//16, num_classes, kernel_size=1))
        


    def forward(self, vis, ir, dsm):


        vis_ir = torch.cat((vis, ir), dim=1)
        b, c, h, w = vis_ir.shape

        vis1, vis2, vis3, vis4 = self.cnn_backbone(vis_ir)

        vis1 = self.vis_dim1(vis1)
        vis2 = self.vis_dim2(vis2)
        vis3 = self.vis_dim3(vis3)
        vis4 = self.vis_dim4(vis4)

        dsm_embed = self.gcn_backbone(dsm)       #[256, 1/16, 1/16]

        vis4, dsm = self.informationChange(vis4, dsm_embed)


        feat = self.encoder(vis4, dsm)

        feat = self.conv1(feat)
        feat = F.interpolate(feat, scale_factor=2, mode='bicubic', align_corners=False)
        dsm = F.interpolate(dsm, scale_factor=2, mode='bicubic', align_corners=False)
        dsm = self.dsm_decoder1(dsm)
        feat = self.ffm1(vis3+dsm, feat)

        feat = self.conv2(feat)
        feat = F.interpolate(feat, scale_factor=2, mode='bicubic', align_corners=False)
        dsm = F.interpolate(dsm, scale_factor=2, mode='bicubic', align_corners=False)
        dsm = self.dsm_decoder2(dsm)
        feat = self.ffm2(vis2+dsm, feat)

        feat = self.conv3(feat)
        feat = F.interpolate(feat, scale_factor=2, mode='bicubic', align_corners=False)
        dsm = F.interpolate(dsm, scale_factor=2, mode='bicubic', align_corners=False)
        dsm = self.dsm_decoder3(dsm)
        feat = self.ffm3(vis1+dsm, feat)

        out = F.interpolate(feat, size=(h, w), mode='bicubic')

        out = self.segmentation_head(out)

        return out








        



