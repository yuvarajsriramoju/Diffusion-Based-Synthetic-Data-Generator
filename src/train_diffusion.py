import argparse, pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from src.utils import set_seed
from src.data import mnist_dataloaders
from src.lit_diffusion import LitDDPM

def main(args):
    set_seed(42)
    train, val = mnist_dataloaders(batch_size=args.batch_size)
    model = LitDDPM(lr=args.lr)
    logger = MLFlowLogger(experiment_name=args.exp, tracking_uri=args.mlflow)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator="auto",
        devices="auto"
    )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
    trainer.save_checkpoint(args.ckpt)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--exp", type=str, default="mnist-ddpm")
    p.add_argument("--mlflow", type=str, default="file:./mlruns")
    p.add_argument("--ckpt", type=str, default="ddpm.ckpt")
    main(p.parse_args())
