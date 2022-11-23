import torch

def mcrmse(y_hat,y):
    mse_per_cols = torch.sqrt(torch.sum(torch.square(y_hat - y),axis=0))
    return torch.mean(mse_per_cols)