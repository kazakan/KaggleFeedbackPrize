import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from transformers import T5Tokenizer

class FeedbackPrizeDataset(Dataset):
    def __init__(
        self,
        file_path,
        tokenizer : None,
        mode : str ='test'
    ):  
        self.tokenizer = tokenizer if tokenizer is not None else T5Tokenizer.from_pretrained("t5-base")
     
        data = pd.read_csv(file_path)
        tokenized = self.tokenizer(data['full_text'].values.tolist(),padding=True,return_tensors="pt")

        self.ids = data['text_id'].values
        self.X = tokenized['input_ids']
        self.attention_masks = tokenized['attention_mask']

        if mode == "train":
            self.y = data.iloc[:,2:].values
        else :
            self.y = np.zeros((len(self.ids),6))

    def __getitem__(self, idx):
        return self.ids[idx], self.X[idx], self.attention_masks[idx],self.y[idx]

    def __len__(self):
        return len(self.ids)


class FeedbackPrizeDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_path = None,
        test_path = None,
        tokenizer = None,
        batch_size=32
    ):
        super().__init__()

        if train_path is None and test_path is None :
            raise ValueError("At least one of train_path or test_path should not be None.")
        
        self.train_path = train_path
        self.test_path = test_path

        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def prepare_data(self):
        if self.train_path is not None:
            self.train_valid_data = FeedbackPrizeDataset(self.train_path,tokenizer=self.tokenizer, mode='train')

        if self.test_path is not None:
            self.test_data = FeedbackPrizeDataset(self.test_path, tokenizer=self.tokenizer)

    def setup(self, stage) -> None:
        if self.train_path is not None and ((stage == "train") or (stage == "fit") or (stage is None)):
            len1 = int(len(self.train_valid_data)*0.8)
            len2 = len(self.train_valid_data) - len1
            self.train_data, self.valid_data = random_split(self.train_valid_data, [len1,len2])

        if self.test_path is not None and ((stage == "test") or (stage is None)):
            pass # self.test_data already prepared

    def train_dataloader(self):
        return DataLoader(self.train_data,shuffle=True,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data,batch_size=self.batch_size)
        