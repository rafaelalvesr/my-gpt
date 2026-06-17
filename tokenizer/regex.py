from collections.abc import Iterable
import multiprocessing
import os
import regex as re

from .base import Tokenizer, freq_pairs, merge_pair


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

_WORKER_STATE = {"pattern": None, "special_tokens": (), "special_pattern": None}


def _build_special_pattern(special_tokens: tuple[str, ...]) -> str | None:
    if not special_tokens:
        return None
    #ensure that longer special tokens are matched before shorter ones
    #  (e.g. <|endoftext|> should be matched before <|endoftext|><|endoftext|>)
    ordered_tokens = sorted(special_tokens, key=len, reverse=True)
    return "(" + "|".join(re.escape(token) for token in ordered_tokens) + ")"


def _split_ordinary_chunks( text: str, special_tokens: tuple[str, ...], special_pattern: str | None,) -> list[str]:
    if not text:
        return []
    if not special_tokens or special_pattern is None:
        return [text]

    parts = re.split(special_pattern, text)
    return [part for part in parts if part and part not in special_tokens]


def _init_pretokenizer_worker(pattern: str, special_tokens: tuple[str, ...]) -> None:
    _WORKER_STATE["pattern"] = re.compile(pattern)
    _WORKER_STATE["special_tokens"] = special_tokens
    _WORKER_STATE["special_pattern"] = _build_special_pattern(special_tokens)


def _pretokenize_batch_worker(batch: str) -> list[list[int]]:
    compiled_pattern = _WORKER_STATE["pattern"]
    if compiled_pattern is None:
        raise RuntimeError("Worker regex pattern is not initialized")

    special_tokens = _WORKER_STATE["special_tokens"]
    special_pattern = _WORKER_STATE["special_pattern"]

    token_sequences = []
    for chunk in _split_ordinary_chunks(batch, special_tokens, special_pattern):
        token_sequences.extend(
            list(match.group().encode("utf-8")) for match in compiled_pattern.finditer(chunk)
        )
    return token_sequences



class RegexTokenizer(Tokenizer):
    """BPE tokenizer with regex pre-tokenization and optional special tokens."""

    def __init__(self, pattern: str | None = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.mapping = {} # Cache the map (token_bytes) -> (token_ids)
        self._chunk_cache = {} # cache bytes -> tuple [int]

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self.special_tokens = dict(special_tokens)
        self.inverse_special_tokens = {
            token_id: token for token, token_id in self.special_tokens.items()
        }

    def _pretokenize(self, text: str) -> list[list[int]]:
        """
        Pre-tokenizes the input text into a list of byte sequences based on the regex pattern and special tokens.
        returns a list of lists of integers, where each inner list corresponds to the byte values of a token.
         For example, if the input text is "Hello, world!" and the pattern splits on words and punctuation, the output 
         might be [[72, 101, 108, 108, 111], [44], [32], [119, 111, 114, 108, 100], [33]] corresponding to the tokens 
         "Hello", ",", " ", "world", and "!".
         If special tokens are present in the text, they will be treated as separate tokens and their byte values will not 
         be included in the output (instead, they will be handled separately in the encode method).
         For example, if the input text is "Hello <|endoftext|> world!" and "<|endoftext|>" is a special token, the output 
         might be [[72, 101, 108, 108, 111], [32], [119, 111, 114, 108, 100], [33]] corresponding to the tokens "Hello", " ", 
         "world", and "!", while the special token "<|endoftext|>" would be handled separately in the encode method.
        """
        special_tokens = tuple(self.special_tokens)
        special_pattern = _build_special_pattern(special_tokens)

        token_sequences = []
        for chunk in _split_ordinary_chunks(text, special_tokens, special_pattern):
            token_sequences.extend(
                list(match.group().encode("utf-8"))
                for match in self.compiled_pattern.finditer(chunk)
            )
        return token_sequences

    def _byte_to_token_id_map(self) -> None:
        if self.mapping:
            return
        for token_id, token_bytes in self.vocab.items():
            if len(token_bytes) == 1:
                self.mapping[token_bytes[0]] = token_id
   

    def _to_base_token_ids(self, byte_values: list[int]) -> list[int]:
        byte_to_token_id = self.mapping
        try:
            return [byte_to_token_id[byte] for byte in byte_values]
        except KeyError as exc:
            raise ValueError(f"Missing base token for byte value {exc.args[0]}") from exc

 
    def _encode_chunk(self, token_ids: list[int]) -> list[int]:
        while len(token_ids) > 1:
            pairs = freq_pairs(token_ids)
            best_pair = min(pairs, key=lambda pair: self.merges.get(pair, float("inf")))
            if best_pair not in self.merges:
                break
            token_ids = merge_pair(token_ids, best_pair, self.merges[best_pair])
        return token_ids

    def encode_ordinary(self, text: str) -> list[int]:
        encoded_ids = []
        for byte_values in self._pretokenize(text):
            key = bytes(byte_values)
            cached = self._chunk_cache.get(key)
            if cached is None:
                base_token_ids = self._to_base_token_ids(byte_values)
                cached = self._encode_chunk(base_token_ids)
                self._chunk_cache[key] = cached
            encoded_ids.extend(cached)
        return encoded_ids

    def _resolve_allowed_special( self,text: str,allowed_special: str | set[str]) -> dict[str, int]:
        if allowed_special == "all":
            return self.special_tokens
        if allowed_special == "none":
            return {}
        if allowed_special == "none_raise":
            if any(token in text for token in self.special_tokens):
                raise ValueError(
                    "Special token found in text, but allowed_special is set to none_raise"
                )
            return {}
        if isinstance(allowed_special, set):
            return {
                token: token_id
                for token, token_id in self.special_tokens.items()
                if token in allowed_special
            }
        raise ValueError(f"allowed_special={allowed_special} not understood")

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        self._byte_to_token_id_map()
        #encode based in: https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
        special_tokens = self._resolve_allowed_special(text, allowed_special)
        
        if not special_tokens:
            return self.encode_ordinary(text)
        

        special_pattern = _build_special_pattern(tuple(special_tokens))
        if special_pattern is None:
            return self.encode_ordinary(text)

        token_ids = []
        for part in re.split(special_pattern, text):
            if not part:
                continue
            if part in special_tokens:
                token_ids.append(special_tokens[part])
            else:
                token_ids.extend(self.encode_ordinary(part))
        return token_ids
    
    def _encode_ordinary_iter(self, text: str) -> Iterable[int]:
        for byte_values in self._pretokenize(text):
            key = bytes(byte_values)
            cached = self._chunk_cache.get(key)
            if cached is None:
                base_token_ids = self._to_base_token_ids(byte_values)
                cached = self._encode_chunk(base_token_ids)
                self._chunk_cache[key] = cached
            yield from cached
    
    def encode_iterable(self, text: Iterable[str], allowed_special: str | set[str] = "all") -> list[int]:
        self._byte_to_token_id_map()
        for batch in text:
            if not batch:
                continue

            special_tokens = self._resolve_allowed_special(batch, allowed_special)
            if not special_tokens:
                yield from self._encode_ordinary_iter(batch)
                continue
            special_pattern = _build_special_pattern(tuple(special_tokens))
            if special_pattern is None:
                yield from self._encode_ordinary_iter(batch)
                continue

            for part in re.split(special_pattern, batch):
                if not part:
                    continue
                if part in special_tokens:
                    yield special_tokens[part]
                else:
                    yield from self._encode_ordinary_iter(part)

    def decode(self, token_ids: list[int]) -> str:
        parts = []
        for token_id in token_ids:
            if token_id in self.vocab:
                parts.append(self.vocab[token_id])
            elif token_id in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[token_id].encode("utf-8"))
            else:
                raise ValueError(f"Token ID {token_id} not found in vocab or special tokens")
        return b"".join(parts).decode("utf-8", errors="replace")

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        token_sequences = self._pretokenize(text)
        if not token_sequences:
            raise ValueError("No valid text found for training.")
        self._run_merge_loop(token_sequences, vocab_size, verbose)

    def train_iterator(self,text: Iterable[str],vocab_size: int,verbose: bool = False) -> None:
        token_sequences = []
        for batch in text:
            if not batch:
                continue
            token_sequences.extend(self._pretokenize(batch))

        if not token_sequences:
            raise ValueError("No valid text batches found in the iterator.")

        self._run_merge_loop(token_sequences, vocab_size, verbose)

    def train_parallel(self,text: Iterable[str],vocab_size: int,verbose: bool = False,num_processes: int | None = None,) -> None:
        if num_processes is None:
            cpu_count = os.cpu_count() or 2
            num_processes = max(1, cpu_count - 1)

        if verbose:
            print(f"Using {num_processes} processes for parallel training")

        token_batches = self.pretokenize_file_parallel(text, num_processes)
        token_sequences = [sequence for batch in token_batches for sequence in batch]

        self._run_merge_loop(token_sequences, vocab_size, verbose)

    def pretokenize_file_parallel(self,iter_text: Iterable[str],num_processes: int) -> list[list[list[int]]]:
        valid_batches = [batch for batch in iter_text if batch]
        if not valid_batches:
            raise ValueError("No valid text batches found in the iterator.")

        if num_processes <= 1:
            return [self._pretokenize(batch) for batch in valid_batches]

        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=num_processes,
            initializer=_init_pretokenizer_worker,
            initargs=(self.pattern, tuple(self.special_tokens)),
        ) as pool:
            return pool.map(_pretokenize_batch_worker, valid_batches)

    def _run_merge_loop(self,token_sequences: list[list[int]],vocab_size: int,verbose: bool = False) -> None:
        assert vocab_size >= 256, (
            "Vocabulary size must be greater than 256"
            " (the number of single byte tokens)."
        )
        if not token_sequences:
            raise ValueError("No valid text batches found in the iterator.")

        n_merges = vocab_size - 256 - len(self.special_tokens)
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        global_pairs, seq_pair_counts, pair_to_seq_ids = self._build_pair_caches(token_sequences)

        for merge_index in range(n_merges):
            if not global_pairs:
                break

            new_token_id = 256 + merge_index
            best_pair = max(
                global_pairs,
                key=lambda pair: (global_pairs[pair], vocab[pair[0]], vocab[pair[1]]),
            )
            affected_seq_ids = list(pair_to_seq_ids.get(best_pair, ()))

            if not affected_seq_ids:
                del global_pairs[best_pair]
                continue

            for seq_id in affected_seq_ids:
                old_local_pairs = seq_pair_counts[seq_id]
                for pair, count in old_local_pairs.items():
                    global_pairs[pair] -= count
                    if global_pairs[pair] <= 0:
                        del global_pairs[pair]

                    seq_ids = pair_to_seq_ids.get(pair)
                    if seq_ids is not None:
                        seq_ids.discard(seq_id)
                        if not seq_ids:
                            del pair_to_seq_ids[pair]

                merged_sequence = merge_pair(token_sequences[seq_id], best_pair, new_token_id)
                token_sequences[seq_id] = merged_sequence

                new_local_pairs = freq_pairs(merged_sequence)
                seq_pair_counts[seq_id] = new_local_pairs
                for pair, count in new_local_pairs.items():
                    global_pairs[pair] = global_pairs.get(pair, 0) + count
                    pair_to_seq_ids.setdefault(pair, set()).add(seq_id)

            vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merges[best_pair] = new_token_id

            if verbose and (merge_index + 1) % 50 == 0:
                print(
                    f"Merge {merge_index + 1}/{n_merges}: "
                    f"Merged pair {best_pair} into token ID {new_token_id}"
                )

        self.merges = merges
        self.vocab = vocab

    def _build_pair_caches(self,token_sequences: list[list[int]]) -> tuple[
        dict[tuple[int, int], int],
        list[dict[tuple[int, int], int]],
        dict[tuple[int, int], set[int]],
    ]:
        global_pairs = {}
        seq_pair_counts = []
        pair_to_seq_ids = {}

        for seq_id, token_ids in enumerate(token_sequences):
            local_pairs = freq_pairs(token_ids)
            seq_pair_counts.append(local_pairs)

            for pair, count in local_pairs.items():
                global_pairs[pair] = global_pairs.get(pair, 0) + count
                pair_to_seq_ids.setdefault(pair, set()).add(seq_id)

        return global_pairs, seq_pair_counts, pair_to_seq_ids
    
    def train_bpe(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the tokenizer using the standar BPE algorithm, without the optimizations in train().
          This is used for testing and comparison purposes.
          """
        assert vocab_size >= 256, "Vocabulary size must be greater than 256 (the number of single byte tokens)."
        n_merges = vocab_size -256 - len(self.special_tokens)
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        pattern_iter = self.compiled_pattern.finditer(text)
        token_sequences = [list(t.group().encode("utf-8")) for t in pattern_iter]

        for i in range (n_merges):
            idx = 256+i
            pairs = {}
            for tokens_ids in token_sequences:
                batch_pairs = freq_pairs(tokens_ids)
                for pair, count in batch_pairs.items():
                    pairs[pair] = pairs.get(pair, 0) + count
            
            if not pairs:
                break
            # we break ties by the order of the tokens in the vocab, favoring pairs with lower token ids
            best_pair,_ = max(pairs.items(), key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]))

            token_sequences = [merge_pair(tokens_ids, best_pair, idx) for tokens_ids in token_sequences]
            vocab[idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merges[best_pair] = idx
        
            if verbose and (i+1) % 50 == 0:
                print(f"Merge {i+1}/{n_merges}: Merged pair {best_pair} into token ID {idx}")
        # Save
        self.merges = merges #use in encode
        self.vocab = vocab #use in decode