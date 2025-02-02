from lib.model import create_trainer
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob

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
            max_steps=200000,
            check_val_every_n_epoch=50,
            enable_progress_bar=False,
            log_every_n_steps=100000,
            num_sanity_val_steps=0
        )
    elif opt.model.mode == '4gpu':
        trainer = pl.Trainer(
            gpus=4,
            strategy='ddp',
            accelerator="gpu",
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback],
            max_steps=200000,
            check_val_every_n_epoch=50,
            logger=logger,
            log_every_n_steps=20,
            num_sanity_val_steps=0
        )

    model = create_trainer(opt)
    checkpoint = sorted(glob.glob("./checkpoints/*.ckpt"))[-1]

    if opt.dataset.testing.type == 'novel_pose':
        testset = create_dataset(opt.dataset.metainfo, opt.dataset.novelpose)
    elif opt.dataset.testing.type == 'novel_view':
        testset = create_dataset(opt.dataset.metainfo, opt.dataset.novelview)
    elif opt.dataset.testing.type == 'canonical':
        testset = create_dataset(opt.dataset.metainfo, opt.dataset.canonical)
    elif opt.dataset.testing.type == 'test':
        testset = create_dataset(opt.dataset.metainfo, opt.dataset.test)

    trainer.test(model, testset, ckpt_path=checkpoint)

if __name__ == '__main__':
    main()