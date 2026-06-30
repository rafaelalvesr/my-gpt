import json
import logging
from dataclasses import asdict
from pathlib import Path
import numpy as np

from train.optimizer import AdamW, learning_rate_schedule, clip_gradients
from train.tensorvalue import no_grad
from model import TransformerLM
from tokenizer import load_gpt2_tokenizer


# Console logging. Per-experiment file logs are attached by the caller (main.py).
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(tokenizer, data_loader, token_path: str):
    """Encode the dataset into tokens and save it as a numpy array"""
    all_ids = []
    for id in tokenizer.encode_iterable(data_loader.iter_chunks()):
        all_ids.append(id)
    
    arr = np.array(all_ids, dtype=np.uint16)
    np.save(token_path,arr)
    logger.info("Save %d tokens in %s", len(arr), token_path)


def get_batch(data: np.ndarray, batch_size: int, seq_len: int):
    """Builds a batch of sequences from the data array. Returns x and y, where y is x shifted by one token."""
    max_start = len(data)-seq_len-1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack( [data[i:i+seq_len].astype(np.uint16) for i in starts])
    y = np.stack([data[i+1:i+seq_len+1].astype(np.uint16) for i in starts])
    return x,y

def evaluate(model, data, num_examples, seq_len):
    """Evaluates the model on a batch of validation data and returns the loss."""
    # Single batched forward over [num_examples, seq_len]; inference only,
    # so run under no_grad: no autograd graph is built (no grad buffers, no cycles).
    x, y = get_batch(data, num_examples, seq_len)
    with no_grad():
        loss = model.loss(x.astype(int), y.astype(int))
    return float(loss.data)


def generate(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generates text from the model given a prompt."""
    tokens = tokenizer.encode(prompt)
    for _ in range(max_length):
        context = tokens[-model.max_seq_len:]
        context = np.array(context, dtype=np.uint16)
        next_id = model.predict_next_token(context, temperature=temperature)
        tokens.append(int(next_id))

    logger.info(tokenizer.decode(tokens))

def save_model(step, model, path, historic):
    """Saves the model and training history to disk."""
    logger.info("Saving model at step %d...", step+1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    base_dir = Path(path).parent
    data_file = base_dir / "history.csv"

    np.savetxt(data_file,
                np.column_stack([historic["loss"], historic["val"]]),
                header="train_loss val_loss")
    model.save(path)
    logger.info("Model and history saved at %s and %s", path, data_file)


def save_trainer(step, optimizer, config):
    """Saves the trainer state to disk."""
    path = Path(config.checkpoint_path).with_name("trainer.npz")
    np.savez(path, 
             step=step, 
             config = json.dumps(asdict(config)), 
             **optimizer.save_dict())
    logger.info("Trainer state saved at %s", path)

def load_trainer(path, optimizer):
    """Loads the trainer state from disk."""
    ck = np.load(path)
    optimizer.load_dict(ck)
    return int(ck['step'])


def train(config):
    """
    Main training loop for the GPT model.
    Notations: batch_size = B, seq_len = S, vocab_size = V, , T = B*S
    """
    # load model from the checkpoint if it exists, otherwise create a new one.
    if Path(config.checkpoint_path).exists():
        logger.info("Loading model from %s...", config.checkpoint_path)
        model = TransformerLM.load(config.checkpoint_path)
    else:
        model = TransformerLM(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_sequence_length,
            d_model=config.model_dim,
            num_heads=config.num_attention_heads,
            d_ff=config.model_dim * 4,  # typically 4 times the model dimension
            num_layers=config.num_layers,
        )

    #Load the optimizer state from the checkpoint if it exists, otherwise create a new one.
    lr_rate = config.learning_rate
    optimizer = AdamW(model.parameters,lr_rate , beta1=config.beta1, beta2=config.beta2, weight_decay=config.weight_decay)
    trainer_path = Path(config.checkpoint_path).with_name("trainer.npz")
    if trainer_path.exists():
        logger.info("Loading trainer state from %s...", trainer_path)
        start_step = load_trainer(trainer_path, optimizer)
    else:
        start_step = 0

    # training history: resume if present, else start fresh (always defined).
    # np.savetxt writes the header as a '#' comment, which np.loadtxt skips by
    # default -> do NOT pass skiprows (it would drop the first data row).
    historic = {"loss": [], "val": []}
    val_media = float("nan")
    history_path = Path(config.checkpoint_path).parent / "history.csv"
    if history_path.exists():
        rows = np.atleast_2d(np.loadtxt(history_path))  # atleast_2d: single-row case
        if rows.size:
            historic = {"loss": rows[:, 0].tolist(), "val": rows[:, 1].tolist()}
            val_media = historic["val"][-1]
    
    # Data training and validation loader
    # mmap_mode='r' maps the file without loading it all into RAM
    data = np.load(config.token_path,mmap_mode='r')
    logger.info("Dataset loaded with %d tokens.", len(data))
    logger.info("Number of batches per epoch: %d", len(data) // (config.batch_size * config.max_sequence_length))
    split = int(config.val_split*len(data))
    train_data, val_data = data[:split], data[split:]
    

    #load tokenizer for generation during training
    tokenizer = load_gpt2_tokenizer()
    
    #print table with training parameters
    row_template = "{:^10} {:^15} {:^15} {:^10} {:^10}"
    logger.info(row_template.format("step", "train_loss", "val_loss", "ppl", "lr"))
    logger.info("-"*60)
    for step in range(start_step, config.num_steps):
        #Learning rate schedule:
        if config.learning_rate_schedule:
            optimizer.lr  = learning_rate_schedule(step=step, T_w= config.warmup_steps, T_c= config.num_steps,
                alpha_max=config.learning_rate,alpha_min=config.learning_rate/10)
        #get batch dataset
        x,y = get_batch(train_data,config.batch_size,config.max_sequence_length) #[B,S]
        optimizer.zero_grad()
        #forward pass
        loss = model.loss(x.astype(int),y.astype(int))
        #backward pass
        loss.backward()
        #gradient clipping:
        if config.clip_gradients: 
            clip_gradients(model.parameters, max_norm=config.max_grad_norm)
        #update weights
        optimizer.step()

        #logging section: calculate and print training loss, validation loss, perplexity, and learning rate.
        #-------------------------------------------------------------------------------------
        loss_media = float(loss.data)
        perplexity = np.exp(min(loss_media, 20))
        historic["loss"].append(loss_media)

        if (step+1) % config.eval_interval==0:
            val_media = evaluate(model, val_data, config.eval_examples, config.max_sequence_length)
        historic["val"].append(val_media)

        logger.info(row_template.format(step+1, f"{loss_media:.3f}", f"{val_media:.3f}", f"{perplexity:.0f}", f"{optimizer.lr:.2e}"))

        if (step+1) % (config.eval_interval)==0:
            logger.info("-"*60)
            generate(model, tokenizer, prompt="Era uma vez", max_length=100, temperature=0.8)
            logger.info("-"*60)
        #-------------------------------------------------------------------------------------
        # Save the model and history every 100 steps and at the end of training.
        if (step+1) % 100 == 0 or (step+1) == config.num_steps:
            save_model(step, model, config.checkpoint_path, historic)
            save_trainer(step+1, optimizer, config)