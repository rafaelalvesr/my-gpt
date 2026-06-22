
from collections.abc import Iterable
from .base import Tokenizer, freq_pairs, merge_pair, naive_bpe

class BasicTokenizer(Tokenizer):
    """Basic tokenizer using BPE algorithm."""
    def __init__(self, pattern: str = None):
        super().__init__()
        self.pattern = pattern

    def train(self, text: str | Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, ("vocab size >= 256 single byte tokens" )
        n_merges = vocab_size - 256
        if isinstance(text, str):
            text = [text]

        token_sequences = [list(b.encode("utf-8")) for b in text if b]
        if not token_sequences:
            raise ValueError("No valid text batches found in the iterator.")
        #interativaely merge the most frequent pairs until we reach the desired vocab size
        self.merges, self.vocab = naive_bpe(token_sequences, n_merges, verbose)

    def encode(self, text: str) -> list[int]:
        byte_text = text.encode("utf-8")
        tokens_ids = list(byte_text)

        while len(tokens_ids) >= 2:
            pairs = freq_pairs(tokens_ids)
            #find the pair with the smallest merge id (most recently merged )
            best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
            idx = self.merges[best_pair]
            tokens_ids = merge_pair(tokens_ids, best_pair, idx)

        return tokens_ids

    def decode(self, token_ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in token_ids).decode("utf-8", errors="replace")