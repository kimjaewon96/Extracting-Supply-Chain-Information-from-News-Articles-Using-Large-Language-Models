import lightning as pl
import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW


class MLP(nn.Module):
    def __init__(self, embedding_dim, seq_len, hidden_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, emb):
        out = self.mlp(emb)
        tag_scores = nn.functional.log_softmax(out, dim=1)
        return tag_scores


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, seq_len, hidden_dim, num_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.linear = nn.Linear(2 * hidden_dim * seq_len, 5)

    def forward(self, emb):
        lstm_out, _ = self.lstm(emb)
        lstm_out = lstm_out.reshape(lstm_out.shape[0], -1)
        # lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear(lstm_out)
        tag_scores = nn.functional.log_softmax(lstm_out, dim=1)
        return tag_scores


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
    ):
        super(BaseModel, self).__init__()
        self.model = None
        self.lr = lr
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        out = self.model(x)
        # out = out.squeeze()
        return out

    def training_step(self, batch, batch_idx):
        x = batch["vector"]
        y = batch["label"]
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["vector"]
        y = batch["label"]
        y_hat = self(x)
        y_pred = y_hat.argmax(dim=1)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        f1 = f1_score(
            y.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average="macro"
        )
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x = batch["vector"]
        y = batch["label"]
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        y_pred = y_hat.argmax(dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        self.test_predictions.extend(y_pred)
        self.test_targets.extend(y)

    def on_test_epoch_end(self):
        predictions = np.array(self.test_predictions)
        targets = np.array(self.test_targets)
        print(classification_report(targets, predictions, digits=3))
        micro_f1 = f1_score(targets, predictions, average="micro")
        macro_f1 = f1_score(targets, predictions, average="macro")
        classwise_f1 = f1_score(targets, predictions, average=None)

        self.log("test_f1_micro", micro_f1)
        self.log("test_f1_macro", macro_f1)
        for i, f1 in enumerate(classwise_f1):
            self.log(f"test_f1_class_{i}", f1)
        self.test_predictions.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class MLPModel(BaseModel):
    def __init__(
        self,
        embedding_dim: int = 300,
        seq_len: int = 380,
        hidden_dim: int = 128,
        lr: float = 1e-4,
    ):
        super(MLPModel, self).__init__(lr=lr)
        self.model = MLP(
            embedding_dim=embedding_dim, seq_len=seq_len, hidden_dim=hidden_dim
        )


class BiLSTMModel(BaseModel):
    def __init__(
        self,
        embedding_dim: int = 300,
        seq_len: int = 380,
        hidden_dim: int = 128,
        lr: float = 1e-4,
    ):
        super(BiLSTMModel, self).__init__(lr=lr)
        self.model = BiLSTM(
            embedding_dim=embedding_dim, seq_len=seq_len, hidden_dim=hidden_dim
        )
