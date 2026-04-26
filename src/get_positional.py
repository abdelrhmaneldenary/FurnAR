import torch
import math

def get_positional_embedding(sequence_length, d):

    positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    
    results = torch.zeros(sequence_length, d)
    
    results[:, 0::2] = torch.sin(positions * div_term)
    
    results[:, 1::2] = torch.cos(positions * div_term)
    
    return results