

from abc import ABC, abstractmethod
from collections.abc import Iterable
import unicodedata

def freq_pairs(token_ids: list[int], pairs_count = None) -> dict[(str, str), int]:
    """
    Computes the frequency of adjacent character pairs in the given text.
    """
    pairs = {} if pairs_count is None else pairs_count
    for p in zip(token_ids, token_ids[1:]):
        pairs[p] = pairs.get(p,0) + 1
    return pairs

def merge_pair(token_ids: list[int], pair: tuple[str,str], new_id: int) -> list[int]:
    """
    Merges the specified pair of characters in the given list of tokens.
    """
    merged = []
    i = 0
    n = len(token_ids)
    while i < n:
        if i < n - 1 and token_ids[i] == pair[0] and token_ids[i+1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(token_ids[i])
            i += 1
    return merged

def naive_bpe(token_sequences: list[list[int]], n_merges: int, verbose=False):
    """Naive BPE: compute the frequency of all pairs from zero at each step, and merge the most frequent pair."""
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    for i in range(n_merges):
        idx = 256 + i
        #global pair frequency, sum all sequences
        pairs = {}
        for ids in token_sequences:
            for pair, count in freq_pairs(ids).items():
                pairs[pair] = pairs.get(pair, 0) + count
        if not pairs:
            break #nothing else to merge
        #most frequent pair, break ties  determinist by bytes of tokens
        best_pair = max(pairs, key=lambda p: (pairs[p], vocab[p[0]], vocab[p[1]]))

        #Apply the merge to all sequences
        token_sequences = [merge_pair(ids, best_pair, idx) for ids in token_sequences]
        vocab[idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = idx
        if verbose and (i+1) % 20 == 0:
            print(f"Merge {i + 1}/{n_merges}: {best_pair} -> {idx}")
        
    return merges, vocab

# Helpers function
def replace_control_characters(s: str) -> str:
    """
        we don't want to print control characters
        which distort the output (e.g. \n or much worse)
        https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        http://www.unicode.org/reports/tr44/#GC_Values_Table
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    """pretty print a token, escaping control characters"""
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer(ABC):
    """
    Abstract base class for tokenizers. 
    Subclasses must implement the train, encode, and decode methods.
    """
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.pattern = "" # regex pattern for tokenization
        self.special_tokens = {} # str -> int
        self.vocab = self.build_vocab() # int -> bytes

    
    def get_ratio(self, text: str, token_ids: list[int]) -> float:
        """Returns the compression ratio of the tokenizer."""
        text_bytes = bytes(text, "utf-8")
        return len(text_bytes) / len(token_ids) if token_ids else 1.0

    @abstractmethod
    def train(self, text: str | Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        """
        Trains the tokenizer on the provided texts and builds a vocabulary of the specified size.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encodes a given text into a list of token IDs.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def build_vocab(self):
        """"Builds the vocabulary mapping from token IDs to byte sequences based on the merges
          and special tokens."""
        
        vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
    
        return vocab
    
    def save(self, prefix: str = "tokenizer") -> None:
        """Saves de vocabulary (.vocab) and parameters(.model) to files with the given prefix."""

        invert_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(f"{prefix}.vocab", "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)

                if idx in invert_merges:
                    p0,p1 = invert_merges[idx]
                    s0 = render_token(self.vocab[p0])
                    s1 = render_token(self.vocab[p1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
                

        with open(f"{prefix}.model", "w", encoding="utf-8") as f:
            #pattern
            f.write(f"pattern\t{self.pattern}\n")
            #special tokens
            for special, idx in self.special_tokens.items():
                f.write(f"special\t{special}\t{idx}\n")

            #Merges
            for pair, idx in self.merges.items():
                f.write(f"merge\t{pair[0]}\t{pair[1]}\t{idx}\n")

    def load(self, model_file: str) -> None:
        "Load the model file"
        assert model_file.endswith(".model")

        merge = {}
        special_tokens = {}

        with open(model_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts)==2 and parts[0] == "pattern":
                    self.pattern = parts[1]
                elif  parts[0] == "special":
                    special_tokens[parts[1]] = int(parts[2])
                elif parts[0] == "merge":
                    merge[(int(parts[1]), int(parts[2]))] = int(parts[3])

        self.merges = merge
        self.special_tokens = special_tokens
        self.vocab = self.build_vocab()


    def visualise_tokens(self,token_values: list[int]) -> None:
        """
        Visualise tokens with different background colors.
        Source: https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
        """
        background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
        # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
        # visualise the token. Here, we'll just use the unicode replacement character to represent some
        # fraction of a character.
        unicode_token_values = [(self.vocab[x]).decode("utf-8", errors="replace") for x in token_values]

        #b''.join(self.vocab[i] for i in token_ids)

        running_length = 0
        last_color = None
        for token in unicode_token_values:
            color = background[running_length % len(background)]
            if color == last_color:
                color = background[(running_length + 1) % len(background)]
                assert color != last_color
            last_color = color
            running_length += len(token)
            print(color + token, end="")
        print("\u001b[0m")
    
    def get_merges_pair(self) -> list[tuple[bytes, bytes]]:
        """Returns a list of merges of string from merges of token ids.
        The purpuse of this is to test the tokenizer using a reference merges (gpt2 merges) that are represented as string pairs, rather than token id pairs.
        """
        merge_texts = []
        for pair, _ in self.merges.items():
            t1 = self.vocab[pair[0]]
            t2 = self.vocab[pair[1]]
            merge_texts.append((t1,t2)) 
        return merge_texts
    
    def get_vocab(self) -> dict[int, bytes]:
        """Build vocab using the actual vocab and the special tokens. 
        The special tokens are added at the beginning of the vocab, 
        so they get the lowest token ids after the single byte tokens.
        The purpuse of this is to test the tokenizer using a reference vocab (gpt2/4 vocab) that has special tokens with low token ids.
        """
        vocab ={}
        for i, special in enumerate(self.special_tokens):
            vocab[i] = special.encode("utf-8")
        
        n_special = len(self.special_tokens)
        #update the vocab with the actual vocab, but shift the token ids by n_special to make room for the special tokens at the beginning
        for idx, token in self.vocab.items():
            vocab[idx + n_special] = token
        
        return vocab