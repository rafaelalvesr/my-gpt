
import os
import codecs
from typing import BinaryIO, Generator, Iterator, Optional
import pyarrow.parquet as pq
import pandas as pd



#source: https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



class ReadTextFile:
    def __init__(self, file_path: str, chunk_size: int = 4*1024*1024, batch_size: int = 4*1024):
        self.file_path = file_path
        self.chunk_size = chunk_size  # 4 MB
        self.batch_size = batch_size  # 4 KB (character count)

    def iter_lines(self, n_lines: Optional[int] = None) -> Generator[str, None, None]:
        """Yields batches of lines up to batch_size characters. Optionally limits to n_lines."""
        batch: list[str] = []
        batch_len = 0
        with open(self.file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if n_lines is not None and i >= n_lines:
                    break
                batch.append(line)
                batch_len += len(line)
                if batch_len >= self.batch_size:
                    yield "".join(batch)
                    batch = []
                    batch_len = 0
        if batch:
            yield "".join(batch)

    def iter_chunks(self) -> Generator[str, None, None]:
        """Yields decoded string chunks of chunk_size bytes. Suitable for large files."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        with open(self.file_path, "rb") as f:
            while True:
                b = f.read(self.chunk_size)
                if not b:
                    break
                yield decoder.decode(b)

    def iter_from_chunks(self, desired_num_chunks: int = 4, split_special_token: bytes =  b"<|endoftext|>") -> Generator[str, None, None]:
        """Yields decoded string chunks of chunk_size bytes, ensuring we don't split in the middle of a character."""
        boundaries = find_chunk_boundaries(open(self.file_path, "rb"), desired_num_chunks=desired_num_chunks, split_special_token=split_special_token)
        with open(self.file_path, "rb") as f:
            for i in range(len(boundaries) - 1):
                f.seek(boundaries[i])
                chunk_size = boundaries[i + 1] - boundaries[i]
                chunk_bytes = f.read(chunk_size)
                yield chunk_bytes.decode("utf-8", errors="ignore")

    def get_all_text(self) -> str:
        """Reads the entire file content. Use only for small files."""
        with open(self.file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def get_text_lines(self, n_lines: int = 1000) -> str:
        """Reads the first n_lines lines. Use only for small/preview reads."""
        lines: list[str] = []
        with open(self.file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for _ in range(n_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
        return "".join(lines)


class ReadParquetFile:
    def __init__(self, file_path: str, batch_size: int = 1000):
        self.file_path = file_path
        self.batch_size = batch_size  # Number of rows per batch

    def get_schema(self) -> pq.ParquetSchema:
        """Returns the Parquet file schema."""
        return pq.read_schema(self.file_path)

    def get_dataframe(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Reads the entire file as a DataFrame. Use only for files that fit in memory."""
        return pq.read_table(self.file_path, columns=columns).to_pandas()

    def iter_chunks(self, columns: Optional[list[str]] = None) -> Iterator[pd.DataFrame]:
        """Yields DataFrames of batch_size rows. Memory-efficient for large files."""
        parquet_file = pq.ParquetFile(self.file_path)
        for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=columns):
            yield batch.to_pandas()

    def iter_column(self, column: str, n_rows: Optional[int] = None) -> Generator[list, None, None]:
        """Yields lists of values from a single column in batches of batch_size rows."""
        parquet_file = pq.ParquetFile(self.file_path)
        total = 0
        for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=[column]):
            values = batch.column(column).to_pylist()
            if n_rows is not None:
                remaining = n_rows - total
                values = values[:remaining]
            yield values
            total += len(values)
            if n_rows is not None and total >= n_rows:
                break

    def iter_lines(self, column: str, n_rows: Optional[int] = None) -> Generator[str, None, None]:
        """Yields concatenated text from a string column, one batch per yield."""
        for values in self.iter_column(column, n_rows=n_rows):
            yield "\n".join(v for v in values if v)

    def get_text_lines(self, column: str, n_rows: int = 1000) -> str:
        """Returns the first n_rows values from a text column joined by newlines."""
        rows: list[str] = []
        for values in self.iter_column(column, n_rows=n_rows):
            rows.extend(v for v in values if v)
        return "\n".join(rows)

