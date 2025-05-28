from transformers import PretrainedConfig

class SmallMindConfig(PretrainedConfig):
    model_type = "smallmind"
    
    def __init__(
        self, 
        max_seq_len: int = 512, 
        batch_size: int = 32,
        n_layer: int = 8,
        n_head: int = 8,
        n_key_value_head: int = 2,
        n_embd: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 6400,
        max_new_tokens: int = 400,
        max_lines: int = 10000,
        norm_eps: float = 1e-5,
        intermediate_size: int = None,
        hidden_act: str = 'silu',
        theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
        use_moe: bool = False,
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
        self.use_moe = use_moe