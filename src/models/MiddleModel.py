import pytorch_lightning as pl
import torch

import pandas as pd
import numpy as np
from datetime import datetime
import pathlib


class MiddleModule(pl.LightningModule):

    """
    Module for code reuse.
    """

    def __init__(self,
            name : str = "noname",
            mean : float = None,
            std : float = None,
        ):
        super().__init__()

        self.name = name
        self.mean = mean
        self.std = std

        self.mseloss = torch.nn.MSELoss()


    def forward(self,batch):
        raise NotImplementedError

    def training_step(self,batch,batch_idx):
        y = batch[-1]

        if self.mean is not None and self.std is not None:
            y = (batch.y - self.mean) / self.std

        y_hat = self(batch)

        loss = self.mseloss(y_hat.squeeze().float(),y.float())
        return loss

    def validation_step(self,batch,batch_idx):
        y = batch[-1]
        
        y_hat = self(batch)

        if self.mean is not None and self.std is not None:
            y_hat = y_hat*self.std + self.mean

        loss = self.mseloss(y_hat.squeeze().float(),y.float())

        return {'MSE':loss.detach(),'n_rows':y_hat.shape[0]}

    def validation_epoch_end(self,outputs):
        sum_square_err = 0
        n_rows = 0
        for output in outputs:
            sum_square_err += (output['MSE'])*output['n_rows']
            n_rows += output['n_rows']

        total_metric = sum_square_err / n_rows

        print(f"Epoch {self.current_epoch} MSE:{total_metric}")
        self.log("MSE",total_metric ,on_step=False, on_epoch=True)

        return {"MSE" : total_metric}

    def test_step(self,batch,batch_idx):
        y_hat = self(batch).detach().cpu().numpy()

        if self.mean is not None and self.std is not None:
            y_hat = y_hat*self.std + self.mean

        return {
            'text_id' : batch[0],
            'yhat' : y_hat.squeeze()
        }
    
    def test_epoch_end(self,outputs):
        text_id = []
        for out in outputs:
            text_id.extend(out['text_id'])

        y_hat = np.concatenate([x['yhat'] for x in outputs],axis=0).squeeze()

        ret = {
            'text_id' :text_id,
            'cohesion':y_hat[:,0],
            'syntax':y_hat[:,1],
            'vocabulary':y_hat[:,2],
            'phraseology':y_hat[:,3],
            'grammar':y_hat[:,4],
            'conventions':y_hat[:,5]
        }

        metadata = {
            'model_name' : self.name,
            'datetime' : datetime.now()
        }

        return ret, metadata

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001, last_epoch=-1)
        
        return [optimizer],[lr_scheduler]
   