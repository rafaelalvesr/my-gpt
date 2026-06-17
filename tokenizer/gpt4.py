#SOURCE:https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py
#I did little changes..
"""
Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
Note that this is a pretrained tokenizer. By default and inside init(), it
loads the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken.
"""

from collections.abc import Iterable
import tiktoken
from .regex import RegexTokenizer
from .base import render_token

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def bpe(mergeable_ranks, token, max_rank):
    """
    helper function used in recover_merges() to reconstruct the merge forest
    """
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    """
    tiktoken gives mergeable_ranks (final token byte sequences + ranks), 
    but not explicit parent pairs.

    the `merges` are already the byte sequences in their merged state.
    so we have to recover the original pairings. We can do this by doing
    a small BPE training run on all the tokens, in their order.
    also see https://github.com/openai/tiktoken/issues/60
    also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306

    - tiktoken gives mergeable_ranks (final token byte sequences + ranks), but not explicit parent pairs.
    """
    merges = {}
    #loops through each token and its rank
    for token, rank in mergeable_ranks.items():
        #skips single-byte tokens (raw base vocabulary)
        if len(token) == 1:
            continue # skip raw bytes
        #calls bpe with max_rank=rank to ask: just before this token’s own merge rank, 
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

class GPT4Tokenizer(RegexTokenizer):
    """Tokenizer using the GPT-4 pattern and special tokens."""
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        # the merges are those of gpt4, but we have to recover them
        self.merges = recover_merges(mergeable_ranks)
        # reconstruct the vocab from the merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # now here is another tricky part.
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here.
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        # finally register the special tokens
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    
    def encode_ordinary(self, text:str) -> list[int]:
        #uodate de the encode to hangle the byte shuffle 
        # (GPT-4 has a weird historical quirk where the single byte tokens are permuted in a different order, 
        # so we have to shuffle the bytes before we encode them, and unshuffle them after we decode them)
        encoded_ids = []

        for byte_values in self._pretokenize(text):
            byte_values_shuffled = [self.byte_shuffle[b] for b in byte_values]
            base_token_ids = self._to_base_token_ids(byte_values_shuffled)
            encoded_ids.extend(self._encode_chunk(base_token_ids))
        return encoded_ids
    
    def decode(self, token_ids: list[int]) -> str:
        # we have to un-permute the bytes before we decode
        text_bytes = b"".join(self.vocab[idx] for idx in token_ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    # this is a pretrained tokenizer, it is not intended to be trained
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    

    # this is a pretrained tokenizer, it is not intended to be trained
    def train_iterator(self, text: Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

 # save/load would require some thought.
    # we'd have to change save/load of base to add support for byte_shuffle...
    # alternatively, we could move byte_shuffle to base class, but that would
    # mean that we're making ugly our beautiful Tokenizer just to support
    # the GPT-4 tokenizer and its weird historical quirks around byte_shuffle.
    def save(self, prefix: str = "tokenizer") -> None:
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file: str) -> None:
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file: str) -> None:
        """         
        just for visualization purposes let's output the GPT-4 tokens
        in the exact same format as the base class would.
        simple run as:
        python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')" 
        """ 
        # build vocab being mindful of the byte shuffle
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # now merge the shuffled bytes and write to file
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")