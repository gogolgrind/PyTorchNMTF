import torch
import torch.nn

def l2(x,y):
    return torch.nn.MSELoss()(x,y)
    
def kl_dev(x,y):
    return (x * torch.log(x/y) - x + y).mean()