import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from math import pi, sqrt, ceil
import numpy as np
import time

from .backbones.convnext import ConvNeXt

class DownConv(nn.Module):
    def __init__(
        self,
        ks,
        ngf,
        new_ngf,
        strd,
        bias=True
    ):
        self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd !=1 else nn.Identity(),
                nn.Conv2d(ngf * strd**2, new_ngf, ks, 1, ceil((ks - 1) // 2), bias=bias)
            )
    
    def forward(self, x):
        return self.downconv(x)

class UpConv(nn.Module):
    def __init__(
        self,
        ks,
        ngf,
        new_ngf,
        strd,
        bias=True   
    ):
        self.upconv = nn.Sequential(
                nn.Conv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=bias),
                nn.PixelShuffle(strd) if strd !=1 else nn.Identity(),
            )
        
    def forward(self, x):
        return self.upconv(x)

class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed
        if 'pe' in pe_embed:
            lbase, levels = [float(x) for x in pe_embed.split('_')[-2:]]
            self.pe_bases = lbase ** torch.arange(int(levels)) * pi

    def forward(self, pos):
        if 'pe' in self.pe_embed:
            value_list = pos * self.pe_bases.to(pos.device)
            pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
            return pe_embed.view(pos.size(0), -1, 1, 1)
        else:
            return pos

class NeRVBlock(nn.Module):
    def __init__(
        self,
        ks,
        ngf,
        new_ngf,
        strd,
        bias=True,
        is_decode=True,
    ):
        super().__init__()
        conv = UpConv if is_decode else DownConv
        self.conv = conv(ngf=ngf, new_ngf=new_ngf, strd=strd, ks=ks, bias=bias)
        self.norm = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class HNeRV(nn.Module):
    def __init__(
        self,
        embed,
        ks,
        num_blks,
        enc_strds,
        enc_dim,
        dec_strds,
        fc_dim,
        reduce,
        lower_width,
    ):
        super().__init__()
        #self.embed = embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in num_blks.split('_')]

        enc_dim1, enc_dim2 = [int(x) for x in enc_dim.split('_')]
        c_in_list, c_out_list = [enc_dim1] * len(enc_strds), [enc_dim1] * len(enc_strds)
        c_out_list[-1] = enc_dim2

        self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=enc_strds, dims=c_out_list, drop_path_rate=0)
        hnerv_hw = np.prod(enc_strds) // np.prod(dec_strds)
        self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
        ch_in = enc_dim2

        decoder_layers = []        
        ngf = fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, ngf=ch_in, new_ngf=out_f, ks=0, strd=1, bias=True)
        decoder_layers.append(decoder_layer1)
        for i, strd in enumerate(dec_strds):                         
            reduction = sqrt(strd) if reduce==-1 else reduce
            new_ngf = int(max(round(ngf / reduction), lower_width))
            for j in range(dec_blks):
                cur_blk = NeRVBlock(dec_block=True, ngf=ngf, new_ngf=new_ngf, ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else strd, bias=True)
                decoder_layers.append(cur_blk)
                ngf = new_ngf
        
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1) 

        self.out = nn.Sigmoid()

    def forward(self, input, input_embed=None):
        if input_embed is not None:
            img_embed = input_embed
        else:
            img_embed = self.encoder(input)
        
        embed_list = [img_embed]
        dec_start = time.time()
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer(output) 
            embed_list.append(output)

        img_out = self.out(self.head_layer(output))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return  img_out, embed_list, dec_time