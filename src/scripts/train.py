import torch
import pytorch_lightning as pl

from transformers import T5Tokenizer
from ..data.dataset import FeedbackPrizeDataModule
from ..models.EncT5 import EncT5MultiRegressModel
from pathlib import Path


if __name__ == "__main__":
    
    pl.seed_everything(42)

    model = EncT5MultiRegressModel()
    
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    datamodule = FeedbackPrizeDataModule(
        Path('./data/train.csv'),Path('./data/test.csv'),
        tokenizer=tokenizer,
        batch_size=1
    )
    trainer = pl.Trainer(precision=32,max_epochs=10)

    trainer.fit(model,datamodule)

    torch.save(model,'./out/model.pt')
    torch.save(tokenizer,'./out/tokenizer.pt')