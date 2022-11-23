from os import PathLike
from typing import Optional
import torch
import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd

from transformers import T5Tokenizer
from ..data.dataset import FeedbackPrizeDataModule
from ..models.EncT5 import EncT5MultiRegressModel

def test(
    test_path : PathLike,
    submission_path : PathLike,
    batch_size : int = 32,
    model_path : Optional[PathLike] = './model.pt',
    tokenizer_path : Optional[PathLike] = './tokenizer.pt',
    acc : str = 'cpu'
):
    pl.seed_everything(42)

    model = torch.load(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

    datamodule = FeedbackPrizeDataModule(
        test_path=test_path,
        tokenizer=tokenizer,
        batch_size=batch_size
    )
    trainer = pl.Trainer(precision=32,accelerator=acc)

    result, metadata = trainer.test(model,datamodule)

    df_result = pd.DataFrame.from_dict(result)
    df_result.to_csv(submission_path, index=False)


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('test_path',type=Path)
    parser.add_argument('submission_path',type=Path)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--model_path',type=Path,default='./model.pt')
    parser.add_argument('--tokenizer_path', type=Path,default='./')
    parser.add_argument('--acc',type=str,default='cpu')

    args = parser.parse_args()

    test(
        args.test_path,
        args.submission_path,
        args.batch_size,
        args.model_path,
        args.tokenizer_path,
        args.acc
    )