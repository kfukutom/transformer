import numpy as np 
import torch
import math
from torch import nn 
import torch.nn.functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

