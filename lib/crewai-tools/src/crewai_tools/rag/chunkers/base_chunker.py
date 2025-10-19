import re


class RecursiveCharacterTextSplitter:
    """A text splitter that recursively splits text based on a hierarchy of separators."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ) -> None:
        """Initialize the RecursiveCharacterTextSplitter.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting (in order of preference)
            keep_separator: Whether to keep the separator in the split text
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) cannot be >= chunk size ({chunk_size})"
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._keep_separator = keep_separator

        self._separators = separators or [
            "\n\n",
            "\n",
            " ",
            "",
        ]

    def split_text(self, text: str) -> list[str]:
        """Split the input text into chunks.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = self._split_text_with_separator(text, separator)

        good_splits = []

        for split in splits:
            if len(split) < self._chunk_size:
                good_splits.append(split)
            else:
                if new_separators:
                    other_info = self._split_text(split, new_separators)
                    good_splits.extend(other_info)
                else:
                    good_splits.extend(self._split_by_characters(split))

        return self._merge_splits(good_splits, separator)

    def _split_text_with_separator(self, text: str, separator: str) -> list[str]:
        if separator == "":
            return list(text)

        if self._keep_separator and separator in text:
            parts = text.split(separator)
            splits = []

            for i, part in enumerate(parts):
                if i == 0:
                    splits.append(part)
                elif i == len(parts) - 1:
                    if part:
                        splits.append(separator + part)
                else:
                    if part:
                        splits.append(separator + part)
                    else:
                        if splits:
                            splits[-1] += separator

            return [s for s in splits if s]
        return text.split(separator)

    def _split_by_characters(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self._chunk_size):
            chunks.append(text[i : i + self._chunk_size])  # noqa: PERF401
        return chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge splits into chunks with proper overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for split in splits:
            split_len = len(split)

            if total + split_len > self._chunk_size and current_doc:
                if separator == "":
                    doc = "".join(current_doc)
                else:
                    if self._keep_separator and separator == " ":
                        doc = "".join(current_doc)
                    else:
                        doc = separator.join(current_doc)

                if doc:
                    docs.append(doc)

                # Handle overlap by keeping some of the previous content
                while total > self._chunk_overlap and len(current_doc) > 1:
                    removed = current_doc.pop(0)
                    total -= len(removed)
                    if separator != "":
                        total -= len(separator)

            current_doc.append(split)
            total += split_len
            if separator != "" and len(current_doc) > 1:
                total += len(separator)

        if current_doc:
            if separator == "":
                doc = "".join(current_doc)
            else:
                if self._keep_separator and separator == " ":
                    doc = "".join(current_doc)
                else:
                    doc = separator.join(current_doc)

            if doc:
                docs.append(doc)

        return docs


class BaseChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ) -> None:
        """Initialize the Chunker.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting
            keep_separator: Whether to keep separators in the chunks
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
        )

    def chunk(self, text: str) -> list[str]:
        """Chunk the input text into smaller pieces.

        Args:
            text: The text to chunk.

        Returns:
            A list of text chunks.
        """
        if not text or not text.strip():
            return []

        return self._splitter.split_text(text)
