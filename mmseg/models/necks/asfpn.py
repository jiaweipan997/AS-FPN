import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer

from mmseg.registry import MODELS

import torch.nn.functional as F
from ..utils import resize
from ..backbones.vit import TransformerEncoderLayer

def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

class ASFModule(nn.Module):
    def __init__(self,
                 in_channels,
                 levels,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 eps=0.0001):
        super(ASFModule, self).__init__()
        self.activation = activation
        self.eps = eps
        self.levels = levels
        self.asfpn_convs = nn.ModuleList()

        # Isotropic Dynamic Fusion
        for i in [2,1,0,1,2]:  
                asfpn_conv = nn.Sequential(
                    ConvModule(
                        in_channels[i],
                        in_channels[i],
                        3,
                        padding=1,
                        groups=in_channels[i],
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=self.activation,
                        inplace=False,
                        ) 
                )
                self.asfpn_convs.append(asfpn_conv)

        ## Isotropic Dynamic Fusion
        self.up_convs = nn.ModuleList()
        for i in range(self.levels-2):
            l_conv = ConvModule(
                in_channels[i],
                in_channels[i+1],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.activation,
                inplace=False,
                )
            self.up_convs.append(l_conv)
        self.down_convs = nn.ModuleList()
        for i in range(self.levels-1):
            l_conv = ConvModule(
                in_channels[i+1],
                in_channels[i],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.activation,
                inplace=False,
                )
            self.down_convs.append(l_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # build top-down
        idx_asfpn = 0
        pathtd = inputs
        pathtd = list(pathtd)
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            pathtd_i = F.interpolate(pathtd[i], scale_factor=2, mode='nearest')
            if pathtd[i - 1].shape[2:] != pathtd_i.shape[2:]:
                pathtd_i = resize(
                            pathtd[i],
                            size=pathtd[i - 1].shape[2:],
                            mode='nearest')
            pathtd[i - 1] = (pathtd[i - 1] + self.down_convs[i-1](pathtd_i))
            pathtd[i - 1] = self.asfpn_convs[idx_asfpn](pathtd[i - 1])
            idx_asfpn = idx_asfpn + 1
        # build down-top
        # Asymmetric Topology against Semantic Dilution
        for i in range(0, levels - 2, 1):
            pathtd_im = F.max_pool2d(pathtd[i], kernel_size=2)
            if pathtd[i + 1].shape[2:] != pathtd_im.shape[2:]:
                pathtd_im = resize(
                            pathtd[i],
                            size=pathtd[i + 1].shape[2:],
                            mode='nearest')
            pathtd[i + 1] = ( pathtd[i + 1] + self.up_convs[i](pathtd_im) +
                              inputs_clone[i + 1])
            pathtd[i + 1] = self.asfpn_convs[idx_asfpn](pathtd[i + 1])
            idx_asfpn = idx_asfpn + 1

        return pathtd

class Asymmetric_Stacked_Fusion(nn.Module):
    def __init__(self,
                 in_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(Asymmetric_Stacked_Fusion, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack

        self.backbone_end_level = self.num_ins
        assert num_outs >= self.num_ins - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.fpn_convs = nn.ModuleList()
        self.stack_asfpn_convs = nn.ModuleList()

        for ii in range(stack):
            self.stack_asfpn_convs.append(ASFModule(in_channels=in_channels,
                                                      levels=self.backbone_end_level-self.start_level,
                                                      conv_cfg=conv_cfg,
                                                      norm_cfg=norm_cfg,
                                                      activation=self.activation
                                                      ))
            
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build top-down and down-top path with stack
        for bifpn_module in self.stack_asfpn_convs:
            inputs = bifpn_module(inputs)

        outs = inputs

        return outs

@MODELS.register_module()
class ASFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 act_cfg=None):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels

        self.Asymmetric_Stacked_Fusion = Asymmetric_Stacked_Fusion(in_channels=self.in_channels, 
                            stack=3,
                            activation=dict(type='ReLU'),
                            num_outs=len(self.in_channels))

        self.attention = TransformerEncoderLayer(
                   embed_dims=self.in_channels[-1],
                   num_heads=4,
                   feedforward_channels=1 * self.in_channels[-1],
                   attn_drop_rate=0.,
                   drop_rate= 0.,
                   drop_path_rate=0.,
                   num_fcs=2,
                   qkv_bias=True,
                   act_cfg=dict(type='GELU'),
                   norm_cfg=dict(type='LN'),
                   with_cp=False,
                   batch_first=True)

        # Backbone-Adaptive Feature Alignment
        self.activate = build_activation_layer(act_cfg)
        

    def forward(self, F_in):
        
        # Global Residual Recursive Learning
        # Holistic Residual Connection
        res_F_in = [F_in.clone() for F_in in F_in]

        # Asymmetric_Stacked_Fusion
        F_stack = self.Asymmetric_Stacked_Fusion(F_in)

        # Semantic Attention
        B, C, H, W = F_stack[-1].shape 
        F_stack[-1] = F_stack[-1].view(B, C, H * W).permute([0, 2, 1]).contiguous()
        F_stack[-1] = self.attention(F_stack[-1])
        F_stack[-1] = F_stack[-1].view(B, H, W, C).permute([0, 3, 1, 2]).contiguous() # F_attn

        # Backbone-Adaptive Feature Alignment
        F_out = [self.activate(i + j) for i, j in zip(F_stack, res_F_in)]

        return tuple(F_out)

