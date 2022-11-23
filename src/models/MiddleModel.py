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

        return {'y_hat':y_hat, 'y':y}

    def validation_epoch_end(self,outputs):
        y_hats = []
        ys = []
        for output in outputs:
            y_hats.append(output['y_hat'])
            ys.append(output['y'])

        y_hats = torch.cat(y_hats,axis=0)
        ys = torch.cat(ys,axis=0)

        mse_per_cols = torch.sqrt(torch.sum(torch.square(y_hats - ys),axis=0))
        mcrmse = torch.mean(mse_per_cols)

        print(f"Epoch {self.current_epoch} MCRMSE:{mcrmse}")
        self.log("MCRMSE",mcrmse ,on_step=False, on_epoch=True)

        return {"MCRMSE" : mcrmse}

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
   