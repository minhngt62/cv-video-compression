import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import torch

from ..models import NeRV, HNeRV
from ..measures import l2, psnr_fn_single
from .compress import quant_model, quant_tensor

class LtHNeRV(L.LightningModule):
    def __init__(
        self,
        fc_dim,
        ks="0_1_5",
        num_blks="1_1",
        enc_strds=[5, 4, 3, 2, 2],
        enc_dim="64_16",
        dec_strds=[5, 4, 3, 2, 2],
        reduce=1.2,
        lower_width=12,

        lr0=0.001,
        betas=(0.9, 0.999),
        weight_decay=0,
        warmup_epochs=30, # 0.2 * 150
        loss_alpha=0.7,
        loss_fn=l2,

        quant_model_bit=8
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = HNeRV(
            fc_dim,
            ks=ks,
            num_blks=num_blks,
            enc_strds=enc_strds,
            enc_dim=enc_dim,
            dec_strds=dec_strds,
            reduce=reduce,
            lower_width=lower_width,
        )

        self.lr0, self.betas, self.weight_decay = lr0, betas, weight_decay
        self.warmup_epochs = warmup_epochs
        self.loss_alpha = loss_alpha
        self.loss_fn = loss_fn
        self.quant_model_bit = quant_model_bit

    def forward(self, img, img_embed=None):
        return self.model(img, img_embed)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), 
            betas=self.betas,
            weight_decay=self.weight_decay,
            lr=self.lr0
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.warmup_epochs, T_mult=1
        )
        return [optimizer], [lr_scheduler]
    
    def _forward_impl(self, batch):
        _, frames = batch
        return self.model(frames), frames

    def _measure(self, pred_frames, frames, mode="train"):
        loss = self.loss_fn(pred_frames, frames)
        psnr = psnr_fn_single(pred_frames.detach(), frames)
        self.log_dict({"%s_loss" % mode: loss, "%s_psnr" % mode: psnr}, prog_bar=True)
        return loss, psnr
    
    def _fps(self, dec_time, batch_size, mode="train"):
        fps = batch_size / dec_time
        self.log_dict({"%s_fps" % mode: fps}, prog_bar=True)
        return fps

    def _quantized_weights(self):
        _, quant_ckt = quant_model(self.model, self.quant_model_bit)
        self.model.load_state_dict(quant_ckt)

    def training_step(self, batch, batch_idx):
        pred_frames, _, dec_time, frames = self._forward_impl(batch)
        loss, _ = self._measure(pred_frames, frames)
        self._fps(dec_time, len(batch))    
        return loss
    
    def on_validation_epoch_start(self):
        self._quantized_weights()
    
    def on_test_epoch_start(self):
        self._quantized_weights()
    
    def validation_step(self, batch, batch_idx):
        pred_frames, _, dec_time, frames = self._forward_impl(batch, mode="valid")
        self._measure(pred_frames, frames)
        self._fps(dec_time, len(batch), "valid")
    
    def test_step(self, batch, batch_idx):
        pred_frames, _, dec_time, frames = self._forward_impl(batch, mode="test")
        self._measure(pred_frames, frames)
        self._fps(dec_time, len(batch), "test")

class LtNeRV(LtHNeRV):
    def __init__(
        self,
        fc_dim,
        embed="pe_1.25_80",
        ks="0_3_3",
        num_blks="1_1",
        dec_strds=[5, 4, 3, 2],
        reduce=2,
        lower_width=12,
        fc_hw="8_16",

        lr0=0.001,
        betas=(0.9, 0.999),
        weight_decay=0,
        warmup_epochs=30, # 0.2 * 150
        loss_alpha=0.7,
        loss_fn=l2,

        quant_model_bit=8
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = NeRV(
            fc_dim=fc_dim,
            embed=embed,
            ks=ks,
            num_blks=num_blks,
            dec_strds=dec_strds,
            reduce=reduce,
            lower_width=lower_width,
            fc_hw=fc_hw,
        )

        self.lr0, self.betas, self.weight_decay = lr0, betas, weight_decay
        self.warmup_epochs = warmup_epochs
        self.loss_alpha = loss_alpha
        self.loss_fn = loss_fn
        self.quant_model_bit = quant_model_bit
    
    def forward(self, img, img_embed=None):
        return self.model(img)