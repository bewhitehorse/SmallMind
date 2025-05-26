from transformers import PretrainedConfig

class SmallMindConfig(PretrainedConfig):
    model_type = "smallmind"
    
    def __init__(
        self, 
        block_size: int = 512, #maximum sequence length
        batch_size: int = 12,
        n_layer: int = 12,
        n_head: int = 12,
        n_key_value_head: int = 6,
        n_embd: int = 768,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        vocab_size: int = 50257,
        max_new_tokens: int = 400,
        max_lines: int = 10000,
        norm_eps: float = 1e-5,
        intermediate_size: int = None,
        hidden_act: str = 'silu',
        theta: float = 10000.0,
        max_position_embeddings: int = 32768,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_key_value_head = n_key_value_head
        self.n_embd = n_embd
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self.max_lines = max_lines
        self.norm_eps = norm_eps
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.theta = theta
        self.max_position_embeddings = max_position_embeddings