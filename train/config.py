

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model training. 
    This class holds all the hyperparameters and settings required for training a GPT model.
    It includes parameters for the model architecture, training process, and data handling. 
    By using a dataclass, we can easily create instances of this configuration and manage our 
    training settings in a structured way.
    """
    # Model parameters
    num_layers: int = 4 #depth of the transformer models
    num_attention_heads: int = 8
    model_dim: int = 256
    max_sequence_length: int = 256
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_steps: int = 10000  #num of batch iteration
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    learning_rate_schedule: bool = True
    clip_gradients: bool = True
    warmup_steps: int = 2 # number of epoch of warmup
    max_grad_norm: float = 1.0 # max grad norm in gradient clipping.
    # Eval parameters:
    eval_interval: int = 10 # validate each N steps
    eval_examples: int = 16 # number of example in evaluation
    val_split: float = 0.95
    # Data parameters - REVISAR
    train_data_path: str = "database/wikipedia_pt_1M.txt" #"database/wikipedia_pt_1M.txt"
    token_path: str = "database/train_data/tokens.npy"
    checkpoint_path: str = "checkpoints/model.npz"
    vocab_size: int = None #update from tokenizer vocabsize

