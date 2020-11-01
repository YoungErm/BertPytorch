import torch
import random
import numpy as np

def setseed(seednum=777):
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    torch.cuda.manual_seed_all(seednum)
    torch.backends.cudnn.deterministic = True
    return