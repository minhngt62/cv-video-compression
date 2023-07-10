import torch
import os
from torchvision.utils import save_image
from torchvision.io import write_video

def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'].to(torch.float32), quant_t['scale'].to(torch.float32)
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t

def setup(
    ckt="checkpoints\\HNeRV\\6M\\ShakeNDry\\hnerv_6M_shakendry_embed.pth", 
    decoder="checkpoints\\HNeRV\\6M\\ShakeNDry\\hnerv_6M_shakendry_decoder.pth", 
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    quant_ckt = torch.load(ckt, map_location="cpu")
    vid_embed = dequant_tensor(quant_ckt['embed']).to(device)
    dequant_ckt = {k: dequant_tensor(v).to(device) for k,v in quant_ckt['model'].items()}
    decoder = torch.jit.load(decoder, map_location='cpu').to(device)
    decoder.load_state_dict(dequant_ckt)

    return decoder, vid_embed

def infer(
    decoder,
    vid_embed,
    frames,
):
    with torch.no_grad():
        img_out = decoder(vid_embed[frames]).cpu()
    return img_out.permute(0, 2, 3, 1)