from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob
from lib.model import create_trainer

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        save_last=True)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.run}")

    if opt.model.mode == '2gpu':
        trainer = pl.Trainer(
            gpus=2,
            strategy='ddp',
            accelerator="gpu",
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback],
            max_steps=opt.model.optimizer.max_step,
            check_val_every_n_epoch=50,
            logger=logger,
            log_every_n_steps=50,
            num_sanity_val_steps=0
        )
    elif opt.model.mode == '4gpu':
        trainer = pl.Trainer(
            gpus=4,
            strategy='ddp',
            accelerator="gpu",
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback],
            max_steps=opt.model.optimizer.max_step,
            check_val_every_n_epoch=50,
            logger=logger,
            log_every_n_steps=50,
            num_sanity_val_steps=0
        )

    model = create_trainer(opt)
    print(model)
    print(f"#parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainset = create_dataset(opt.dataset.metainfo, opt.dataset.train)
    validset = create_dataset(opt.dataset.metainfo, opt.dataset.valid)

    if opt.model.is_continue == True:
        checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
    else:
        trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    main()