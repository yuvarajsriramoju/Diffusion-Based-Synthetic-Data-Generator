import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

class LitClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SmallCNN()
        self.acc = Accuracy(task="multiclass", num_classes=10)
    def training_step(self, batch, _):
        x,y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits,y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, _):
        x,y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits,y)
        acc = self.acc(logits.softmax(-1), y)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
