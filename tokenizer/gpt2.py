"""
This module contains the implementation of the GPT-2 tokenizer.
We load the pre-trained GPT-2 tokenizer vocab and merges files, and convert them into a format that can be used by our RegexTokenizer implementation.
The GPT-2 tokenizer uses a byte-level BPE encoding, which means that it operates on bytes rather than unicode characters. This allows it to handle any 
text without needing a fixed vocabulary of unicode characters.
"""
from __future__ import annotations
from functools import lru_cache
import json
import os
#import resource

from tokenizer import RegexTokenizer , GPT2_SPLIT_PATTERN

FIXTURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../database/tokenizer/")
VOCAB_PATH = os.path.join(FIXTURES_PATH, "gpt2_vocab.json")
MERGES_PATH = os.path.join(FIXTURES_PATH, "gpt2_merges.txt")

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def get_path(file_name: str) -> str:
    return os.path.join(FIXTURES_PATH, file_name)



def get_tokenizer(vocab, merges, special_tokens):
    tokenizer = RegexTokenizer(pattern=GPT2_SPLIT_PATTERN)
    tokenizer.register_special_tokens(special_tokens)
    tokenizer.vocab = vocab
    tokenizer.merges = merges
    return tokenizer

def load_gpt2_tokenizer(
    vocab_path: str | os.PathLike = VOCAB_PATH,
    merges_path: str | os.PathLike = MERGES_PATH,
    special_tokens: list[str] | None = None
):
    """
    Loads the GPT-2 tokenizer from the given vocab and merges files, and returns a RegexTokenizer instance with the loaded vocab and merges.
    """
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, "r", encoding="utf-8") as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = [] #list of tuples of the form [(merge_token_1, merge_token_2), ...]
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    dict_special_tokens = {} #special token format to RegexTokenizer.register_special_tokens is {special_token: token_id, ...}
    inv_vocab = {v: k for k, v in vocab.items()} #helper dict byte -> token_id
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token
                dict_special_tokens[special_token] = len(vocab)-1
            else:
                dict_special_tokens[special_token] = inv_vocab[byte_encoded_special_token]

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]

    #convert merges from [(merge_token_1, merge_token_2), ...] to {(merge_token_1, merge_token_2): idx, ...}
    

    new_merges = {} # (int, int) -> int
    for merge_token_1, merge_token_2 in merges:
        merge_token_1_id = inv_vocab[merge_token_1]
        merge_token_2_id = inv_vocab[merge_token_2]
        token_1 = vocab[merge_token_1_id]
        token_2 = vocab[merge_token_2_id]
        merge_token = token_1 + token_2
        new_merges[(merge_token_1_id, merge_token_2_id)] = inv_vocab[merge_token]

    return get_tokenizer(vocab, new_merges, dict_special_tokens)
