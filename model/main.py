from attrs import Required

required = Required()
required.importPackages()
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import logging
import functools

#global
MAX = 5000

def device_wrapper(func):

    @functools.wraps(func)
    def wrapper_get_device(*args, **kwargs):
        try:
            print("Retrieving device . . .")
            device = func(*args, **kwargs)
            print(f"Device retrieved: {device}")
            return device
        except Exception as e:
            print(f"Error in wrapper_get_device at {e}")
            sys.exit(1)

    return wrapper_get_device

@device_wrapper
def get_device():
    """
    Retrieve the available device (CUDA if Available, else CPU)
    Args: Requires None
    """
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Usage of the get_device wrapper func.
device = get_device()   
print(f"Active Device: {device}\n")    


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2) / math.sqrt(d_k))
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
    
    def forward(self):
        try:
            even_i = torch.arange(0, self.d_model, 2).float()
            denom = torch.pow(10000, even_i / self.d_model)
            position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
            even_PE = torch.sin(position / denom)
            odd_PE = torch.cos(position / denom)
            merged_PE = torch.stack([even_PE, odd_PE], dim=2)
            PE = torch.flatten(merged_PE, start_dim=1, end_dim=2)
            return PE
        
        except Exception as e:
            logging.error(f"Error in PositionalEncoding forward pass: {e}")
            raise RuntimeError(f"Positional Encoding encountered an error.") from e

# Calling on the positional encoder
pe = PositionalEncoding(d_model=6, max_sequence_length=40)
pe.forward()

class LayerNormalization(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self):
        try:
            out = self.d_model ** 2
            return out
        
        except Exception as e:
            logging.error(f"Error in LayerNormalization forward pass: {e}")
            raise RuntimeError("\nLayerNormalization encountered an error.") from e


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"value.size(): {values.size()}, attention.size(): {attention.size()}")
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")
        out = self.linear_later(values)
        print(f"out.size(): {out.size()}")
        return out # eof
