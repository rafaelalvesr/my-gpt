

from collections.abc import Iterable

from .base import Tokenizer


class CharTokenizer(Tokenizer):
    """
    Represent a string as s sequence of Unicode code points
    """
    def __init__(self):
        super().__init__()
        # the char tokenizer is just a simple mapping of each byte to a token id

    def encode(self, text: str) -> list[int]:
        return list(map(ord, text))
    
    def decode(self, token_ids: list[int]) -> str:
        return "".join(map(chr, token_ids))
    
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        # the char tokenizer does not need to be trained, since it is just a simple mapping of each byte to a token id
        pass


class ByteTokenizer(Tokenizer):
    """
    Represent a string as s sequence of bytes
    """
    def __init__(self):
        super().__init__()
        # the byte tokenizer is just a simple mapping of each byte to a token id

    def encode(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        token_ids = list(map(int, text_bytes))
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")
    
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        # the byte tokenizer does not need to be trained, since it is just a simple mapping of each byte to a token id
        pass
    
