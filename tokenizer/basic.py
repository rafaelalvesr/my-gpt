
from collections.abc import Iterable
from .base import Tokenizer, freq_pairs, merge_pair

class BasicTokenizer(Tokenizer):
    """Basic tokenizer using BPE algorithm."""
    def __init__(self, pattern: str = None):
        super().__init__()
        self.pattern = pattern


    def train_iterator(self, text: Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, (
            "Vocabulary size must be greater than 256"
              " (the number of single byte tokens)."
        )
        n_merges = vocab_size - 256

        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        #iterate over the text iterator and build the initial token ids list
        token_batches = []
        for batch in text:
            if not batch:
                continue
            tokens_ids = list(batch.encode("utf-8"))
            token_batches.append(tokens_ids)
        
        if not token_batches:
            raise ValueError("No valid text batches found in the iterator.")

        #interativaely merge the most frequent pairs until we reach the desired vocab size
        for i in range (n_merges):
            idx = 256 +i
            pairs = {}
            for tokens_ids in token_batches:
                batch_pairs = freq_pairs(tokens_ids)
                for pair, count in batch_pairs.items():
                    pairs[pair] = pairs.get(pair, 0) + count

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            token_batches = [
                merge_pair(tokens_ids, best_pair, idx) 
                for tokens_ids in token_batches
            ]
            vocab[idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merges[best_pair] = idx

            if verbose and (i+1) % 10 == 0:
                print(f"Merge {i+1}/{n_merges}: Merged pair {best_pair} into token ID {idx}")
        # Save
        self.merges = merges #use in encode
        self.vocab = vocab #use in decode

            
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "Vocabulary size must be greater than 256 (the number of single byte tokens)."
        n_merges = vocab_size - 256

        byte_text = text.encode("utf-8")
        tokens_ids = list(byte_text)

        #There are 256 possible byte values, so we start with a vocab of 256 single byte tokens
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        #interativaely merge the most frequent pairs until we reach the desired vocab size
        for i in range (n_merges):
            idx = 256 +i
            pairs = freq_pairs(tokens_ids)
            if not pairs:
                break
            best_pair = max(pairs, key=lambda p: (pairs.get(p, -1), p))
            tokens_ids = merge_pair(tokens_ids, best_pair, idx)
            vocab[idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merges[best_pair] = idx

            if verbose and (i+1) % 10 == 0:
                print(f"Merge {i+1}/{n_merges}: Merged pair {best_pair} into token ID {idx}")
        
        # Save
        self.merges = merges #use in encode
        self.vocab = vocab #use in decode

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