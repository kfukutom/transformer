import pandas as pd
import torch
from torch.nn import nn
import torch.nn.functional as F
from sklearn.preprocessing import minmax_scale
import numpy as np

class Optimizer():
    def __init__(self, params: str, d_t:int, q, k, v):
        super().__init__()
        self.params = params
        self.d_t = d_t
        self.q = q
        self.k = k
        self.v = v
    
    class WordEmbedder():
        def __init__(self, optimizer_instance, word: str):
            super().__init__()
            self.optimizer_instance = optimizer_instance
            self.word = word
        
        def forward(self):
            return np.matmul(self.optimizer_instance.q / self.optimnizer_instance.k ** 2)

optimizer = Optimizer(params="some_params", d_t=100, q=1, k=2, v=3)
word_embedder = optimizer.WordEmbedder(optimizer_instance=optimizer, word="My name is Ken")
word_embedder.forward()