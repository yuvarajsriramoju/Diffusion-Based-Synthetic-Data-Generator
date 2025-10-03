import argparse, torch, pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path
import random

from src.lit_classifier import LitClassifier

def mnist_loaders(batch_size=256, num_workers=2, subset_size=None):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("data/mnist", train=True, download=True, transform=tf)
    if subset_size is not None and subset_size < len(train):
        idx = torch.randperm(len(train))[:subset_size]
        train.data = train.data[idx]
        train.targets = train.targets[idx]
    test = datasets.MNIST("data/mnist", train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    )

class ImageFolderGray(Dataset):
    def __init__(self, root):
        self.paths = sorted(Path(root).glob("*.png"))
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = default_loader(str(self.paths[i]))
        return self.tf(x), 0  # unlabeled synthetic

class MixedDataset(Dataset):
    """Real labeled MNIST + unlabeled synthetic with simple mixup to borrow real labels."""
    def __init__(self, mnist_ds, synth_root, synth_ratio=5.0):
        self.mnist = mnist_ds
        self.synth = ImageFolderGray(synth_root)
        self.len_synth = min(int(len(self.mnist)*synth_ratio), len(self.synth))
    def __len__(self): return len(self.mnist) + self.len_synth
    def __getitem__(self, idx):
        if idx < len(self.mnist):
            return self.mnist[idx]
        xs, _ = self.synth[random.randrange(len(self.synth))]
        xr, yr = self.mnist[random.randrange(len(self.mnist))]
        lam = 0.7
        x = lam*xr + (1-lam)*xs
        return x, yr

def main(args):
    train_loader, val_loader = mnist_loaders(
        batch_size=args.batch_size,
        num_workers=2,
        subset_size=args.subset_size
    )
    if args.mode == "real":
        train_ds = train_loader.dataset
    else:
        train_ds = MixedDataset(train_loader.dataset, args.synthetic_dir, synth_ratio=args.synth_ratio)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = LitClassifier(lr=args.lr)
    logger = MLFlowLogger(experiment_name=args.exp, tracking_uri=args.mlflow)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator="auto",
        devices="auto",
        precision="16-mixed"  # AMP
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_loader)
    res = trainer.validate(model, dataloaders=val_loader)
    print(res)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["real","mix"], default="real")
    p.add_argument("--synthetic_dir", type=str, default="synthetic/mnist_ddpm")
    p.add_argument("--synth_ratio", type=float, default=5.0)   # e.g., 10k real + 50k synth
    p.add_argument("--subset_size", type=int, default=10000)   # low-data regime for the boost
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--exp", type=str, default="mnist-classifier")
    p.add_argument("--mlflow", type=str, default="file:./mlruns")
    args = p.parse_args()
    main(args)
