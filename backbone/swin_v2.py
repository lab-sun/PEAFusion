# Written by Yan Wang based on the following repositories.
# Swin-Transformer: https://github.com/microsoft/Swin-Transformer
# Mask2Former: https://github.com/facebookresearch/Mask2Former
# CRM_RGBTSeg: https://github.com/UkcheolShin/CRM_RGBTSeg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


class Channel_Projector(nn.Module):
    def __init__(self, rgb_input_channel=3, thermal_input_channel=1, expect_channel =3):
        super().__init__()

        self.rgb_input_channel = rgb_input_channel
        self.thermal_input_channel = thermal_input_channel
        self.expect_channel = expect_channel

        self.thermal_channel_align = nn.Conv2d(in_channels=self.thermal_input_channel, out_channels=self.expect_channel, kernel_size=(1,1))
        # Initialize the convolutional layer with Kaiming initialization
        nn.init.kaiming_normal_(self.thermal_channel_align.weight, mode='fan_out')
        if self.thermal_channel_align.bias is not None:
            nn.init.constant_(self.thermal_channel_align.bias, 0)

    def forward(self, x, mode:str):
        if mode == 'rgb':
            x = x
        elif mode == 'thermal':
            x = self.thermal_channel_align(x)
        else:
            print("The input must be rgb or thermal!!!")    
        return x
    
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, identical_connection=False):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.with_iden_con = identical_connection

        self.adapter_gate = nn.Parameter(torch.tensor(1e-6))

    def forward(self, x):

        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.with_iden_con is not True:
            return xs * self.adapter_gate
        else:
            return xs * self.adapter_gate + x

    
class Multi_Scale_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.125, act_layer=nn.GELU, br_grps = 1):    
        super(Multi_Scale_Adapter, self).__init__()

        self.D_features = D_features
        self.hidden_dim = int(D_features*mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, self.hidden_dim)
        self.act = act_layer()
        self.D_fc2 = nn.Linear(self.hidden_dim, D_features)

        self.act_ = act_layer()
        self.inter_dim = int(self.hidden_dim /4)       

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size=3, stride=1, padding=1,groups=br_grps),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size=3, stride=1, padding=1,groups=br_grps),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size=5, stride=1, padding=2,groups=br_grps),
        )
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(self.inter_dim, self.inter_dim, kernel_size=7, stride=1, padding=3,groups=br_grps),
        )
        self.branches = nn.ModuleList(
            [self.branch1x1, self.branch3x3, self.branch5x5, self.branch7x7]
        )

        self.mv_ad_gate = nn.Parameter(torch.tensor(1e-6))

        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # Apply Xavier initialization for other Conv2d layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, H,W):

        B, N, _  = x.shape

        x_down = self.D_fc1(x)
        x_down = self.act(x_down)   # shape: (B,N,C*mlp_raito)
        # split the x_down into 4 chunks, shape: (B,N, C*mlp_ratio*0.25)
        x_down_list = torch.chunk(x_down, 4, dim=-1)

        features = []
        for i_feature in range(len(x_down_list)):
            feature = x_down_list[i_feature].reshape(B, H, W, -1).permute(0, 3, 1, 2)
            feature = self.branches[i_feature](feature)
            feature = feature.permute(0,2,3,1).reshape(B,N,-1)
            features.append(feature)
        x = torch.cat(features, dim=-1)
        
        x = self.act_(x)
        x = self.D_fc2(x)
        
        return x * self.mv_ad_gate
        
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
        output_attn (bool): If True, return the attention map. Default: False
    """

    def __init__(self, dim, window_size, num_heads, 
                 qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0], 
                 output_attn = False,
                 ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.adapter_mod_1_bias = nn.Parameter(torch.ones((num_heads,  window_size[0]*window_size[1], window_size[0]*window_size[1])), requires_grad=True)
        self.adapter_mod_2_bias = nn.Parameter(torch.zeros((num_heads, window_size[0]*window_size[1], window_size[0]*window_size[1])), requires_grad=True)      
        self.output_attn =  output_attn

    def forward(self,x1, x2, mask=None):
        assert x1.shape == x2.shape , "The shape of RGB input and thermal input must be same."
        B_, N, C = x1.shape
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv1 = F.linear(input=x1, weight=self.qkv.weight, bias=qkv_bias)      
        qkv1 = qkv1.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

        qkv2 = F.linear(input=x2, weight=self.qkv.weight, bias=qkv_bias)   
        qkv2 = qkv2.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)     
        q2, k2, v2 = qkv2[0], qkv1[1], qkv2[2]

        q = torch.cat([q1,q2], dim= -2)
        k = torch.cat([k1,k2], dim= -2)
        v = torch.cat([v1,v2], dim= -2)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()

        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

         
        relative_position_bias_0 = torch.cat([relative_position_bias+ 2 * self.adapter_mod_1_bias, relative_position_bias+self.adapter_mod_2_bias], dim=-1)
        relative_position_bias_1 = torch.cat([relative_position_bias+ 2 * self.adapter_mod_1_bias, relative_position_bias+self.adapter_mod_2_bias], dim=-1)
        relative_position_bias_ = torch.cat([relative_position_bias_0, relative_position_bias_1], dim=-2)

        attn = attn + relative_position_bias_.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]  # (nW, N, N)
            mask_0 = torch.cat([mask,mask], dim=-1)
            mask_1 = torch.cat([mask,mask], dim=-1)
            mask_ = torch.cat([mask_0, mask_1], dim=-2)
            attn = attn.view(B_ // nW, nW, self.num_heads, 2*N, 2*N) + mask_.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads,2*N, 2*N)
            attn = self.softmax(attn)

        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_ = (attn @ v).transpose(1, 2).reshape(B_, 2*N, C)

        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)

        x1_ = x_[:,:N,:].contiguous()
        x2_ = x_[:,N:,:].contiguous()

        if self.output_attn:
            return x1_, x2_, attn
        else:
            return x1_, x2_


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.

    """
    def __init__(self, 
                 dim, 
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 pretrained_window_size=0,
                 add_block_adapter = True, 
                 adapter_scale = 4.0, 
                 mha_adapter_ratio = 0.125, 
                 mha_adapter_groups = 1,
                 ffn_adapter_ratio = 0.5,
                 blk_output_attn = False,
                   ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.H = None
        self.W = None

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            output_attn = blk_output_attn
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # conv adapter
        if add_block_adapter:
            self.scale = adapter_scale
            self.add_block_adapter = add_block_adapter
            # The multi-vew adapter for attention module.
            self.attn_conv_adapter_1 = Multi_Scale_Adapter(D_features=dim,mlp_ratio= mha_adapter_ratio, br_grps = mha_adapter_groups)
            self.attn_conv_adapter_2 = Multi_Scale_Adapter(D_features=dim,mlp_ratio= mha_adapter_ratio, br_grps = mha_adapter_groups)
            # The adapter for feedforward network.
            self.adapter_bypass_1 = Adapter(D_features=dim, mlp_ratio =ffn_adapter_ratio, identical_connection=False)
            self.adapter_bypass_2 = Adapter(D_features=dim, mlp_ratio =ffn_adapter_ratio, identical_connection=False)

        else:
            self.scale = None
            self.add_block_adapter = None
            self.attn_conv_adapter_1 = None
            self.attn_conv_adapter_2 = None
            self.ffn_conv_adapter_1 = None
            self.ffn_conv_adapter_2 = None

        self.blk_output_attn = blk_output_attn

    def forward(self, x1, x2):

        H, W = self.H, self.W
        assert x1.shape == x2.shape,  "The shape of RGB input and thermal input must be same."
        B, L, C = x1.shape
        assert L == H * W, "input feature has wrong size"

        shortcut1 = x1
        x1 = x1.view(B, H, W, C)
        shortcut2 = x2
        x2 = x2.view(B, H, W, C)    

        # zero-padding to make the input could be partitioned
        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        x1 = F.pad(x1, (0, 0, 0, pad_r, 0, pad_b))
        x2 = F.pad(x2, (0, 0, 0, pad_r, 0, pad_b))

        assert x1.shape == x2.shape, "two branch of padded input feature must have same shape!"
        B, pad_H, pad_W, C = x1.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x2 = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # calculate attention mask for SW-MSA
            H, W = self.H, self.W
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            H_, W_ = H+pad_b,W+pad_r 
            img_mask = torch.zeros((1, H_, W_, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            assert x1.device == x2.device
            attn_mask = attn_mask.to(x1.device)
        else:
            shifted_x1 = x1
            shifted_x2 = x2
            attn_mask = None

        # partition windows
        x1_windows = window_partition(shifted_x1, self.window_size)  # nW*B, window_size, window_size, C
        x2_windows = window_partition(shifted_x2, self.window_size)  # nW*B, window_size, window_size, C

        x1_windows = x1_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x2_windows = x2_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.blk_output_attn:
            attn1_windows, attn2_windows, attn_maps = self.attn(x1_windows,x2_windows, mask=attn_mask)  #attn1_windows: (nW*B, window_size*window_size, C); attn_maps: (nW*B,num_heads,2*window_size*window_size, 2*window_size*window_size)
            # merge windows
            attn1_windows = attn1_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x1 = window_reverse(attn1_windows, self.window_size, pad_H, pad_W)  # B H' W' C

            attn2_windows = attn2_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x2 = window_reverse(attn2_windows, self.window_size, pad_H, pad_W)  # B H' W' C
            # output window-based attention maps
            attn_maps = attn_maps.view(B,pad_H // self.window_size, pad_W // self.window_size,self.num_heads, 2*self.window_size*self.window_size, 2*self.window_size*self.window_size)
            # assume B=1 when testing for simplicity  & the attention map of center window   & the first head
            attn_maps = attn_maps[0, (pad_H // self.window_size) //2, (pad_W // self.window_size)//2, 0]  # shape (2*window_size*window_size, 2*window_size*window_size)

            # reverse cyclic shift
            if self.shift_size > 0:
                x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                x2 = torch.roll(shifted_x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x1 = shifted_x1
                x2 = shifted_x2
        else:
            attn1_windows, attn2_windows = self.attn(x1_windows,x2_windows, mask=attn_mask)
            # merge windows
            attn1_windows = attn1_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x1 = window_reverse(attn1_windows, self.window_size, pad_H, pad_W)  # B H' W' C

            attn2_windows = attn2_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x2 = window_reverse(attn2_windows, self.window_size, pad_H, pad_W)  # B H' W' C
            # reverse cyclic shift
            if self.shift_size > 0:
                x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                x2 = torch.roll(shifted_x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x1 = shifted_x1
                x2 = shifted_x2

        # unpad the paded feature map
        x1 = x1[:,:H, :W, :].contiguous()
        x2 = x2[:,:H, :W, :].contiguous()  

        x1 = x1.view(B, H * W, C)
        x2 = x2.view(B, H * W, C)

        if self.add_block_adapter:
            x1 = shortcut1 + self.drop_path(self.norm1(x1 + self.scale* self.attn_conv_adapter_1(shortcut1,H,W) ))
            x2 = shortcut2 + self.drop_path(self.norm1(x2 + self.scale* self.attn_conv_adapter_2(shortcut2,H,W) ))
            #FFN
            x1_ = x1 + self.drop_path(self.norm2(  self.mlp(x1) + self.scale* self.adapter_bypass_1(x1) ))
            x2_ = x2 + self.drop_path(self.norm2(  self.mlp(x2) + self.scale* self.adapter_bypass_2(x2) ))
        else:
            x1 = shortcut1 + self.drop_path(self.norm1(x1))
            x2 = shortcut2 + self.drop_path(self.norm1(x2))
            #FFN
            x1_ = x1 + self.drop_path(self.norm2(  self.mlp(x1)  ))
            x2_ = x2 + self.drop_path(self.norm2(  self.mlp(x2)  ))

        if self.blk_output_attn:
            return x1_ , x2_, attn_maps     # attn_maps: (2*window_size*window_size, 2*window_size*window_size)
        else:
            return x1_ , x2_
    
class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
            if the in_chans is 3, process as usual,
            elif in_chans is 1, use conv 1x1 to project the channel number to 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.patches_resolution = None
        self.num_patches = None

    def forward(self, x):
        #padding
        _, _ ,H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        self.patches_resolution = [] 

        x = self.proj(x)
        Wh, Ww = x.size(2), x.size(3)
        self.patches_resolution = [ Wh//self.patch_size[0], Ww//self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] 

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1,2)
            x = self.norm(x)
            x = x.transpose(1,2).view(-1, self.embed_dim, Wh,Ww)

        return x
    
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
        """
    def __init__(self, 
                 dim, 
                 depth, 
                 num_heads, 
                 window_size,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 norm_layer=nn.LayerNorm, 
                 downsample=None, 
                 use_checkpoint=False,
                 pretrained_window_size=0,
                 add_layer_adapter = True,
                 layer_adapter_scale = 4.0,
                 layer_mha_ratio = 0.125,
                 layer_mha_groups = 1, 
                 layer_ffn_ratio = 0.5,
                 layer_output_attn = False
                ):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 add_block_adapter=add_layer_adapter,
                                 adapter_scale= layer_adapter_scale,
                                 mha_adapter_ratio= layer_mha_ratio,
                                 mha_adapter_groups = layer_mha_groups,
                                 ffn_adapter_ratio = layer_ffn_ratio,
                                 blk_output_attn = layer_output_attn
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.layer_output_attn = layer_output_attn

    def forward(self, x1, x2, H, W):

        if self.layer_output_attn:
            attn_map_list = []  # element in attn_map_list has shape : (2*window_size*window_size, 2*window_size*window_size)
            for blk in self.blocks:
                blk.H, blk.W = H, W
                if self.use_checkpoint:
                    x1, x2, attn_maps = checkpoint.checkpoint(blk, x1, x2)
                    attn_map_list.append(attn_maps)
                else:
                    x1, x2, attn_maps = blk(x1, x2)
                    attn_map_list.append(attn_maps)
            if self.downsample is not None:
                x1_ = self.downsample(x1, H, W)
                x2_ = self.downsample(x2, H, W)
                Wh, Ww = (H+1) //2, (W+1)//2
                return x1, x2, H, W,  x1_, x2_, Wh,Ww, attn_map_list
            return x1, x2, H, W, x1, x2, H,W, attn_map_list
        else:
            for blk in self.blocks:
                blk.H, blk.W = H, W
                if self.use_checkpoint:
                    x1, x2 = checkpoint.checkpoint(blk, x1, x2)
                else:
                    x1, x2 = blk(x1, x2)
            
            if self.downsample is not None:
                x1_ = self.downsample(x1, H, W)
                x2_ = self.downsample(x2, H, W)
                Wh, Ww = (H+1) //2, (W+1)//2

                return x1, x2, H, W,  x1_, x2_, Wh,Ww
            return x1, x2, H, W, x1, x2, H,W,
    
    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

class Fusion_Swin_Transformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrained(str): path to load pretrained weights 
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        frozen_stages: layer number to freeze the parameter when training. Default: 4
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (list): Whether to use checkpointing on certain layer to save memory. Default: [False,False,False,False]
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """
    def __init__(
        self, 
        pretrained=None, 
        patch_size=4, 
        in_chans=3, 
        frozen_stages= 4,         
        embed_dim=96, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24],
        window_size=7, 
        mlp_ratio=4., 
        qkv_bias=True,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, 
        ape=False, 
        patch_norm=True,
        use_checkpoint_list=[False,False,False,False], 
        pretrained_window_sizes=[0, 0, 0, 0],
        add_model_adapter = True,
        model_adapter_scale = 4.0,
        model_mha_ratio = 0.125,
        model_mha_groups = 1,
        model_ffn_ratio = 0.5,
        model_output_attn = False,
        **kwargs):
        super().__init__()


        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        print(self.embed_dim)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = [int(embed_dim * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.mlp_ratio = mlp_ratio

        self.frozen_stages = frozen_stages
        self.pretrained = pretrained

        self.channel_projector = Channel_Projector()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding-> default: False
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        assert len(use_checkpoint_list) == len(depths)
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint= use_checkpoint_list[i_layer],
                               pretrained_window_size=pretrained_window_sizes[i_layer],
                               add_layer_adapter= add_model_adapter,
                               layer_adapter_scale= model_adapter_scale,
                               layer_mha_ratio=model_mha_ratio,
                               layer_mha_groups = model_mha_groups,
                               layer_ffn_ratio= model_ffn_ratio,
                               layer_output_attn = model_output_attn
                               )
            self.layers.append(layer)

        self.model_output_attn = model_output_attn

        self.feature_norms = nn.ModuleList(nn.LayerNorm(dimension) for dimension in self.num_features)


        self.init_weights()
        self._freeze_stages()
                    
    def _freeze_stages(self):

        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for name,param in self.patch_embed.named_parameters(): 
                param.requires_grad = False             

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for name,param in m.named_parameters():
                    if 'adapter'  in name :   
                        param.requires_grad = True
                    else:
                        param.requires_grad = False


    def init_weights(self):

        def _init_weights(m):

            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(self.pretrained, str):
            self.apply(_init_weights)

            print("load model from: ", self.pretrained)

            checkpoint = torch.load(self.pretrained)
            state_dict = checkpoint['model']

            # Define a list of substrings to exclude
            substrings_to_exclude = ['attn_mask', 'relative_coords_table', 'relative_position_index', 'head']

            # Create a new state_dict containing only the parameters you want to keep
            filtered_state_dict = {k: v for k, v in state_dict.items() if not any(substring in k for substring in substrings_to_exclude)}
            self.load_state_dict(filtered_state_dict, strict=False)
            print("Load pretrained weights successfully !!!")

            del checkpoint
            torch.cuda.empty_cache()

        elif self.pretrained is None:
            self.apply(_init_weights)
            print("Load weights from nowhere!!!")
        else:
            raise TypeError('pretrained must be a str or None')


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}
    
    def forward(self, x1, x2):
        
        out_features = {}
        attention_maps = {}

        x1 = self.channel_projector(x1, 'rgb')
        x2 = self.channel_projector(x2, 'thermal')

        x1 = self.patch_embed(x1)   # shape: (B, embed_dim , Wh,Ww)
        x2 = self.patch_embed(x2)

        Wh, Ww = x1.size(2), x1.size(3)

        if self.ape:
            x1 = x1 + self.absolute_pos_embed
            x2 = x2 + self.absolute_pos_embed

        x1 = x1.flatten(2).transpose(1,2)
        x2 = x2.flatten(2).transpose(1,2)
        
        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)

        sim_loss = 0. 

        if self.model_output_attn:
            attention_maps = []
            for i_layer in range(self.num_layers):

                x1_, x2_,H, W, x1, x2, Wh,Ww, attn_map_list = self.layers[i_layer](x1, x2, Wh, Ww)

                out = torch.max(x1_, x2_)
                out = self.feature_norms[i_layer](out)
                out = out.view(-1, H, W, self.num_features[i_layer] ).permute(0, 3, 1, 2).contiguous()
                
                out_features["res{}".format(i_layer + 2)] = out
                attention_maps.append(attn_map_list)

            flat_list = [tensor.unsqueeze(0) for sublist in attention_maps for tensor in sublist] 
            # Use torch.cat to concatenate along dim=0.
            attn_result = torch.cat(flat_list, dim=0) # shape: (depth, 2*win*win, 2*win*win)

            return out_features,  attn_result
        else:
            for i_layer in range(self.num_layers):

                x1_, x2_,H, W, x1, x2, Wh,Ww = self.layers[i_layer](x1, x2, Wh, Ww)

                out = torch.max(x1_, x2_)
                out = self.feature_norms[i_layer](out)
                out = out.view(-1, H, W, self.num_features[i_layer] ).permute(0, 3, 1, 2).contiguous()
                
                out_features["res{}".format(i_layer + 2)] = out

            return out_features

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(Fusion_Swin_Transformer, self).train(mode)
        self._freeze_stages()

@BACKBONE_REGISTRY.register()
class RGBTSwinTransformer(Fusion_Swin_Transformer, Backbone):
    def __init__(self, cfg, input_shape):

        pretrained= cfg.MODEL.SWIN.PRETRAINED
        patch_size=4
        in_chans=3
        frozen_stages= cfg.MODEL.SWIN.FROZEN_STAGE        
        embed_dim= cfg.MODEL.SWIN.EMBED_DIM
        depths= cfg.MODEL.SWIN.DEPTHS 
        num_heads= cfg.MODEL.SWIN.NUM_HEADS
        window_size= cfg.MODEL.SWIN.WINDOW_SIZE
        mlp_ratio=4.
        qkv_bias=True
        drop_rate=0. 
        attn_drop_rate=0.
        drop_path_rate= cfg.MODEL.SWIN.DROP_PATH_RATE
        norm_layer=nn.LayerNorm
        ape=False
        patch_norm=True
        use_checkpoint_list= cfg.MODEL.SWIN.USE_CHECKPOINT_LIST
        pretrained_window_sizes= cfg.MODEL.SWIN.PRETRAINED_WINDOW_SIZE
        add_model_adapter = cfg.MODEL.SWIN.ADD_MODEL_ADAPTER
        model_adapter_scale = cfg.MODEL.SWIN.MODEL_ADAPTER_SCALE
        model_mha_ratio = cfg.MODEL.SWIN.MODEL_MHA_RATIO
        model_mha_groups = cfg.MODEL.SWIN.MODEL_MHA_GROUPS
        model_ffn_ratio = cfg.MODEL.SWIN.MODEL_FFN_RATIO
        model_output_attn = cfg.MODEL.SWIN.MODEL_OUTPUT_ATTN
        

        super().__init__(
            pretrained, 
            patch_size, 
            in_chans, 
            frozen_stages,         
            embed_dim, 
            depths, 
            num_heads,
            window_size, 
            mlp_ratio, 
            qkv_bias,
            drop_rate, 
            attn_drop_rate, 
            drop_path_rate,
            norm_layer, 
            ape, 
            patch_norm,
            use_checkpoint_list, 
            pretrained_window_sizes,
            add_model_adapter,
            model_adapter_scale,
            model_mha_ratio,
            model_mha_groups,
            model_ffn_ratio,
            model_output_attn,
        )

        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

        self.model_output_attn = model_output_attn

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"

        rgb_input = x[:,:3]
        thermal_input = x[:,3:]
        outputs = {}

        if self.model_output_attn is not True:
            out_features = super().forward(rgb_input, thermal_input)
            align_dict = {}
            return out_features ,  align_dict
        else:
            out_features, attention_maps = super().forward(rgb_input, thermal_input)
            return out_features, attention_maps    

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

