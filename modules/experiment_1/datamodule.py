import os
import urllib.request
import zipfile
from pathlib import Path

import lightning as L
import torch
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize as tokenizer
from torch.utils.data import DataLoader

from datasets import disable_caching, load_from_disk

disable_caching()


def vectorize(row, emb_model, seq_len):
    row["vector"] = torch.zeros(seq_len, 300)
    for idx, token in enumerate(tokenizer(row["masked_text"])):
        try:
            vector = torch.tensor(emb_model[token])
        except KeyError:
            vector = torch.zeros(300)
        if token == "__NE_FROM__":
            vector[-3:] = torch.tensor([1, 1, 0])
        elif token == "__NE_TO__":
            vector[-3:] = torch.tensor([1, 0, 1])
        elif token == "__NE_OTHER__":
            vector[-3:] = torch.tensor([0, 1, 1])
        row["vector"][idx] = vector
    return row


class GLoVEDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, seq_len: int):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data = None
        self.emb_model = None

    def prepare_data(self):
        self.dataset = load_from_disk("../../datasets/ManualDataset")
        self.dataset = self.dataset.select_columns(["masked_text", "label"])
        glove_file = "glove.840B.300d.txt"
        if not self.emb_model:
            if not Path(glove_file).is_file():
                print("Downloading GLoVE...")
                urllib.request.urlretrieve(
                    "http://nlp.stanford.edu/data/glove.840B.300d.zip",
                    "glove.840B.300d.zip",
                )
                with zipfile.ZipFile("glove.840B.300d.zip", "r") as zip_ref:
                    zip_ref.extractall(".")
                os.remove("glove.840B.300d.zip")
            print("Loading GLoVE...")
            self.emb_model = KeyedVectors.load_word2vec_format(
                glove_file, no_header=True, binary=False
            )

    def setup(self, stage: str = "fit"):
        keys = ["train", "valid"] if stage == "fit" else ["test"]
        for split in keys:
            self.dataset[split] = self.dataset[split].map(
                vectorize,
                fn_kwargs={"emb_model": self.emb_model, "seq_len": self.seq_len},
                batched=False,
                remove_columns=["masked_text"],
            )
            self.dataset[split].set_format(type="torch")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["valid"],
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            # num_workers=1,
            # pin_memory=True,
            # persistent_workers=True,
        )
