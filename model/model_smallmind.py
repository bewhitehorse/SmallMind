import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_smallmind import SmallMindConfig

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_dim))
        self.eps = config.norm_eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6):
    freqs = 1 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def rotate_half(x):
    return torch.cat((-x[..., x.size(-1) // 2:], x[..., :x.size(-1) // 2]), dim=-1)

def apply_rotary_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep: int):
    batch_size, seq_len, n_key_value_head, head_size = x.size()
    if n_rep == 1:
        return x
    return (
        x[:,:,:,None,:]
        .expand(batch_size, seq_len, n_key_value_head, n_rep, head_size)
        .reshape(batch_size, seq_len, n_key_value_head * n_rep, head_size)
    )

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_key_value_head = config.n_head if config.n_key_value_head is None else config.n_key_value_head
        assert self.n_head % self.n_key_value_head == 0 
        self.head_size = config.hidden_dim // self.n_head
        self.n_rep = self.n_head // self.n_key_value_head
        self.q_proj = nn.Linear(config.hidden_dim, self.head_size * self.n_head, bias = False)
        self.k_proj = nn.Linear(config.hidden_dim, self.head_size * self.n_key_value_head, bias = False)
        self.v_proj = nn.Linear(config.hidden_dim, self.head_size * self.n_key_value_head, bias = False)
        self.o_proj = nn.Linear(self.n_head * self.head_size, config.hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, 
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                use_cache = False,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, hidden_size = x.size()
        xq = self.q_proj(x) 
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(batch_size, seq_len, self.n_head, self.head_size)
        xk = xk.view(batch_size, seq_len, self.n_key_value_head, self.head_size)
        xv = xv.view(batch_size, seq_len, self.n_key_value_head, self.head_size)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xq], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores + torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = scores @ xv
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_size)
        output = self.output_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int ((config.hidden_dim * 4) * (2/3))
            config.intermediate_size = 64 * ((intermediate_size+64-1) // 64)
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias = False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias = False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_dim, bias = False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
    
class MoEGate(nn.Module):
    pass

class MOEFeedForward(nn.Module):
    pass
 
class SmallMindBlock(nn.Module):
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.attn_norm = RMSNorm(config)
        self.ffn_norm = RMSNorm(config)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                use_cache: Optional[bool] = False, 
                attention_mask: Optional[torch.Tensor] = None):
        
        # Apply the attention
        residual = hidden_states
        hidden_states, present_key_value = self.attn(
            self.attn_norm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states = residual + hidden_states

        # Apply the feed forward network
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states, present_key_value

class SmallMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim = config.hidden_dim // config.n_head, 
                                                    end = config.max_position_embeddings, 
                                                    theta = config.theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.blocks = nn.Sequential(
            *[SmallMindBlock(l,config) for l in range(config.n_layer)]
        )
        self.ln_final = RMSNorm(config)
        self.dropout = nn.Dropout(config.dropout)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                idx: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                targets=None,
                **kwargs):
        batch_size, seq_len = idx.size()
        past_key_values = past_key_values or [None] * len(self.blocks)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.token_embedding_table(idx))

        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + seq_len],
            self.freqs_sin[start_pos: start_pos + seq_len]
        )

        presents = []
        for layer_id, (block, past_key_value) in enumerate(zip(self.blocks, past_key_values)):
            hidden_states, present = block(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.ln_final(hidden_states)

        aux_loss = sum(
            block.mlp.aux_loss
            for block in self.blocks
            if isinstance(block.ffn, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss

class SmallMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SmallMindConfig

    def __init__(self, config: SmallMindConfig = None):
        self.config = config or SmallMindConfig()
        super().__init__(self.config)
        self.model = SmallMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)
        self.model.token_embedding_table.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self, 
                idx: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            idx = idx,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            use_cache = use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT