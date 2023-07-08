import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from typing import Any, Dict, Optional, Union
import os
from copy import copy
from pathlib import Path
PRJ_ROOT = Path(__file__).parent.parent.parent.resolve()


def train(
    model,
    dataset,
    
    default_root_dir=os.path.join(PRJ_ROOT, "configs", "nerv"),
    max_epochs=150,
    log_every_n_steps=30,
    check_val_every_n_epoch=30,
    
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    drop_last=False
):
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        callbacks=[
            ModelCheckpoint(mode="max", monitor="val_psnr"),
            LearningRateMonitor("epoch")
        ],
        default_root_dir=default_root_dir,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch
    )
    trainer.logger._log_graph = False  # if True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # optional logging argument that we don't need

    # build data loaders
    train_loader = data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_memory, 
        drop_last=drop_last
    )
    val_loader = data.DataLoader(
        dataset, shuffle=False, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_memory, 
        drop_last=drop_last
    )

    L.seed_everything(42)
    trainer.fit(model, train_loader, val_loader)
    return trainer.checkpoint_callback.best_model_path

def test(
    model,
    dataset,
    weights,
    
    default_root_dir=os.path.join(PRJ_ROOT, "configs", "nerv"),
    max_epochs=1,
    log_every_n_steps=1,
    
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    drop_last=False
):
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        default_root_dir=default_root_dir,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
    )

    trainer.logger._log_graph = False  # if True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # optional logging argument that we don't need

    val_loader = data.DataLoader(
        dataset, shuffle=False, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_memory, 
        drop_last=drop_last
    )

    model = model.load_from_checkpoint(weights)
    test_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    return test_result