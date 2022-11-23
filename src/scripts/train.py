from os import PathLike
from typing import Optional
import torch
import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser

from transformers import T5Tokenizer
from ..data.dataset import FeedbackPrizeDataModule
from ..models.EncT5 import EncT5MultiRegressModel

from pytorch_lightning.loggers import TensorBoardLogger

def train(
    train_path : PathLike,
    max_epochs : int = 10,
    batch_size : int = 32,
    model_save_path : Optional[PathLike] = './model.pt',
    tokenizer_save_path : Optional[PathLike] = './tokenizer.pt',
    acc : str = 'cpu'
):
    pl.seed_everything(42)

    model = EncT5MultiRegressModel()
    
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    datamodule = FeedbackPrizeDataModule(
        train_path=train_path,
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    logger = TensorBoardLogger(save_dir="./tb_logs",name=model.name)
    trainer = pl.Trainer(precision=32,max_epochs=max_epochs,accelerator=acc,logger=logger)

    trainer.fit(model,datamodule)

    torch.save(model,model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('train_path',type=Path)
    parser.add_argument('--max_epochs',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--model_save_path',type=Path,default='./model.pt')
    parser.add_argument('--tokenizer_save_path', type=Path,default='./')
    parser.add_argument('--acc',type=str,default='cpu')

    args = parser.parse_args()

    train(
        args.train_path,
        args.max_epochs,
        args.batch_size,
        args.model_save_path,
        args.tokenizer_save_path,
        args.acc
    )