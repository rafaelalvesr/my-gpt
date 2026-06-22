"""
Simplified, self-sufficient test suite for the tokenizers.

Covers, with minimal redundancy:
  1. BPE training correctness (deterministic Wikipedia example)
  2. encode/decode round-trip identity (Basic & Regex)
  3. GPT-2 parity with tiktoken (via the package's `load_gpt2_tokenizer`)
  4. special-token handling
  5. GPT-4 parity with tiktoken (cl100k)
  6. reference BPE training (merges/vocab match a known-good snapshot)
  7. save / load round-trip
  8. streaming `encode_iterable`

Fixtures live in `database/tokenizer/`. Tests that need `tiktoken` are skipped
automatically when it is not installed (via `pytest.importorskip`).

Run:  pytest test/test_tokenizer.py -v
"""
from __future__ import annotations

import json
import os

import pytest

from tokenizer import BasicTokenizer, RegexTokenizer, GPT4Tokenizer, GPT2_SPLIT_PATTERN
from tokenizer import load_gpt2_tokenizer
from tokenizer.gpt2 import gpt2_bytes_to_unicode
from data.prepare import ReadTextFile

# --------------------------------------------------------------------------- #
# Paths & helpers
# --------------------------------------------------------------------------- #
TOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "tokenizer")


def fixture(name: str) -> str:
    return os.path.join(TOK_DIR, name)


def read(name: str) -> str:
    with open(fixture(name), encoding="utf-8") as f:
        return f.read()


# A handful of representative strings exercise the tricky paths.
SAMPLE_STRINGS = [
    "",                              # empty
    "s",                             # single ascii char
    "🙃",                            # single unicode char (multi-byte)
    "Hello, how are you?",           # ascii words + punctuation
    "Héllò hôw are ü? 🙃",           # mixed unicode + emoji
]


@pytest.fixture(scope="module")
def gpt2():
    """A GPT-2 tokenizer loaded from database/tokenizer/, with <|endoftext|> registered."""
    return load_gpt2_tokenizer(special_tokens=["<|endoftext|>"])


# --------------------------------------------------------------------------- #
# 1. BPE training correctness (no external deps)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [BasicTokenizer, RegexTokenizer])
def test_bpe_wikipedia_example(factory):
    """
    Wikipedia BPE example: "aaabdaaabac" with 3 merges -> [258, 100, 258, 97, 99]
    (a=97 b=98 c=99 d=100; Z=aa=256, Y=ab=257, X=ZY=258).
    """
    text = "aaabdaaabac"
    expected = [258, 100, 258, 97, 99]

    tok = factory()
    tok.train(text, 256 + 3)
    assert tok.encode(text) == expected
    assert tok.decode(tok.encode(text)) == text

    # training from an iterable of texts must produce the same result
    tok = factory()
    tok.train([text], 256 + 3)
    assert tok.encode(text) == expected


# --------------------------------------------------------------------------- #
# 2. encode/decode round-trip identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [BasicTokenizer, RegexTokenizer])
@pytest.mark.parametrize("text", SAMPLE_STRINGS)
def test_roundtrip_identity(factory, text):
    tok = factory()
    tok.train("the quick brown fox jumps over the lazy dog 1234567890", 256 + 20)
    assert tok.decode(tok.encode(text)) == text


# --------------------------------------------------------------------------- #
# 3. GPT-2 parity with tiktoken
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", SAMPLE_STRINGS + ["__german__", "__tinystories__"])
def test_gpt2_matches_tiktoken(gpt2, text):
    enc = pytest.importorskip("tiktoken").get_encoding("gpt2")
    if text == "__german__":
        text = read("german.txt")
    elif text == "__tinystories__":
        text = read("tinystories_sample.txt")

    # the tinystories sample contains <|endoftext|>; allow it so both encoders
    # treat it as a special token (a no-op for the other strings).
    reference_ids = enc.encode(text, allowed_special={"<|endoftext|>"})
    ids = gpt2.encode(text)
    assert ids == reference_ids
    assert gpt2.decode(ids) == text
    assert enc.decode(reference_ids) == text


# --------------------------------------------------------------------------- #
# 4. special-token handling
# --------------------------------------------------------------------------- #
def test_gpt2_special_tokens(gpt2):
    enc = pytest.importorskip("tiktoken").get_encoding("gpt2")
    text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    ids = gpt2.encode(text)
    # the special token survives as its own piece and is not split
    pieces = [gpt2.decode([i]) for i in ids]
    assert pieces.count("<|endoftext|>") == 3
    assert gpt2.decode(ids) == text

    # and it matches tiktoken when special tokens are allowed
    assert ids == enc.encode(text, allowed_special={"<|endoftext|>"})


def test_overlapping_special_tokens():
    tok = load_gpt2_tokenizer(special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    ids = tok.encode(text)
    pieces = [tok.decode([i]) for i in ids]
    # the longer special token wins where the two overlap
    assert pieces.count("<|endoftext|><|endoftext|>") == 1
    assert pieces.count("<|endoftext|>") == 1
    assert tok.decode(ids) == text


# --------------------------------------------------------------------------- #
# 5. GPT-4 parity with tiktoken (cl100k)
# --------------------------------------------------------------------------- #
GPT4_SPECIALS = (
    "<|endoftext|>Hello world\n"
    "<|fim_prefix|>has<|fim_suffix|> tokens<|fim_middle|> FIM<|endofprompt|>"
)


@pytest.mark.parametrize("text", SAMPLE_STRINGS + [GPT4_SPECIALS])
def test_gpt4_matches_tiktoken(text):
    enc = pytest.importorskip("tiktoken").get_encoding("cl100k_base")
    tok = GPT4Tokenizer()
    assert tok.encode(text, allowed_special="all") == enc.encode(text, allowed_special="all")


# --------------------------------------------------------------------------- #
# 6. reference BPE training: merges & vocab match a known-good snapshot
# --------------------------------------------------------------------------- #
def test_train_bpe_matches_reference():
    tok = RegexTokenizer(pattern=GPT2_SPLIT_PATTERN)
    tok.register_special_tokens({"<|endoftext|>": 0})
    text = ReadTextFile(fixture("corpus.en")).get_all_text()
    tok.train(text, vocab_size=500, verbose=False)

    byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with open(fixture("train-bpe-reference-merges.txt"), encoding="utf-8") as f:
        reference_merges = [
            (bytes(byte_decoder[c] for c in a), bytes(byte_decoder[c] for c in b))
            for a, b in (line.rstrip().split(" ") for line in f if line.strip())
        ]
    assert tok.get_merges_pair() == reference_merges

    with open(fixture("train-bpe-reference-vocab.json"), encoding="utf-8") as f:
        ref_vocab = {
            idx: bytes(byte_decoder[c] for c in item)
            for item, idx in json.load(f).items()
        }
    vocab = tok.get_vocab()
    assert set(vocab.keys()) == set(ref_vocab.keys())
    assert set(vocab.values()) == set(ref_vocab.values())


def test_train_bpe_special_token_not_merged():
    """A registered special token must never be merged into other vocab entries."""
    tok = RegexTokenizer(pattern=GPT2_SPLIT_PATTERN)
    tok.register_special_tokens({"<|endoftext|>": 0})
    text = read("tinystories_sample.txt")
    tok.train(text, vocab_size=512, verbose=False)

    for word in tok.get_vocab().values():
        if word != b"<|endoftext|>":
            assert b"<|" not in word


# --------------------------------------------------------------------------- #
# 7. save / load round-trip
# --------------------------------------------------------------------------- #
def test_save_load(tmp_path):
    text = read("tinystories_sample.txt")
    tok = RegexTokenizer()
    tok.train(text, 256 + 64)
    tok.register_special_tokens({"<|endoftext|>": 256 + 64})
    ids = tok.encode(text, "all")
    assert tok.decode(ids) == text

    prefix = str(tmp_path / "tok")
    tok.save(prefix)

    reloaded = RegexTokenizer()
    reloaded.load(prefix + ".model")
    assert reloaded.encode(text, "all") == ids
    assert reloaded.decode(ids) == text


# --------------------------------------------------------------------------- #
# 8. streaming encode_iterable
# --------------------------------------------------------------------------- #
def test_encode_iterable_roundtrip(gpt2):
    path = fixture("tinystories_sample.txt")
    with open(path, encoding="utf-8") as f:
        ids = list(gpt2.encode_iterable(f))
    assert gpt2.decode(ids) == read("tinystories_sample.txt")

# --------------------------------------------------------------------------- #
# 9. Parallel training 
# --------------------------------------------------------------------------- #
def test_train_parallel_matches_serial():
    text = ReadTextFile(fixture("corpus.en")).get_all_text()
    batches = [text[i:i + 256] for i in range(0, len(text), 256)]
    assert len(batches) > 1  

    serial = RegexTokenizer(pattern=GPT2_SPLIT_PATTERN)
    parallel = RegexTokenizer(pattern=GPT2_SPLIT_PATTERN)

    vocab_size = 320  # 64 merges; 
    serial.train(batches, vocab_size=vocab_size, verbose=False)
    parallel.train_parallel(
        batches,
        vocab_size=vocab_size,
        verbose=False,
        num_processes=2,
    )

    assert parallel.merges == serial.merges
    assert parallel.vocab == serial.vocab
    assert parallel.encode(text) == serial.encode(text)
    assert parallel.decode(parallel.encode(text)) == text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
