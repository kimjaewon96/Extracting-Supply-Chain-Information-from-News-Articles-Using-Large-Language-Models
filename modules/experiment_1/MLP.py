import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from datamodule import GLoVEDataModule
from models import BiLSTMModel, MLPModel

pl.seed_everything(42)

seq_len = 380
datamodule = GLoVEDataModule(batch_size=32, seq_len=seq_len)


def train(mode="MLP"):
    if mode == "MLP":
        model = MLPModel(hidden_dim=128, seq_len=seq_len, lr=0.0005)
    elif mode == "BiLSTM":
        model = BiLSTMModel(hidden_dim=16, seq_len=seq_len, lr=0.0005)
    logger = CSVLogger("lightning_logs", name=f"iitp_{mode.lower()}")
    checkpoint_callback = ModelCheckpoint(
        # monitor="val_f1",
        save_top_k=1,
        save_last=False,
        # mode="max",
        save_weights_only=True,
    )
    # Early stopping: patience회 이상 validation loss가 개선되지 않으면 종료
    # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],  # , early_stopping_callback
        max_epochs=25,
    )
    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Evaluate the model on the test set
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    train("MLP")
    train("BiLSTM")
