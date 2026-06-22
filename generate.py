import argparse

from model import TransformerLM
from tokenizer import load_gpt2_tokenizer

parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
parser.add_argument("--model", default="checkpoints/model.npz",
                    help="path to the model.npz (e.g. runs/lr_3e-3/model.npz)")
parser.add_argument("--prompt", default="Era uma vez", help="prompt to start from")
parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
parser.add_argument("--max-new-tokens", type=int, default=100, help="number of tokens to generate")
args = parser.parse_args()

tokenizer = load_gpt2_tokenizer()
model = TransformerLM.load(args.model)

tokens = tokenizer.encode(args.prompt)

for _ in range(args.max_new_tokens):
    context = tokens[-model.max_seq_len:]
    next_id = model.predict_next_token(context, temperature=args.temperature)
    tokens.append(int(next_id))

print(tokenizer.decode(tokens))
