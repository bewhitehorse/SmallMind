import torch
from dataclasses import dataclass

@dataclass
class SmallMindConfig:
    block_size: int = 512 #maximum sequence length
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    n_key_value_head: int = 6
    n_embd: int = 768
    hidden_dim: int = n_embd
    dropout: float = 0.1
    vocab_size: int = 50257
    max_new_tokens: int = 400
    max_lines: int = 10000
    norm_eps: float = 1e-5
    intermediate_size: int = None
    hidden_act: str = 'silu'
    theta: float = 10000.0
    max_position_embeddings: int = 32768