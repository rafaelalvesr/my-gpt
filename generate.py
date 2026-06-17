from model import TransformerLM
from tokenizer import load_gpt2_tokenizer

tokenizer =  load_gpt2_tokenizer()
model = TransformerLM.load("./checkpoints/model.npz")

prompt= "Era uma vez"
tokens = tokenizer.encode(prompt)

for _ in range(100):
    context = tokens[-model.max_seq_len:]
    next_id = model.predict_next_token(context, temperature = 0.8)
    tokens.append(int(next_id))

print(tokenizer.decode(tokens))
