"""Constants for file handling utilities."""

from datetime import timedelta
from typing import Final, Literal


DEFAULT_MAX_FILE_SIZE_BYTES: Final[Literal[524_288_000]] = 524_288_000
MAGIC_BUFFER_SIZE: Final[Literal[2048]] = 2048

UPLOAD_MAX_RETRIES: Final[Literal[3]] = 3
UPLOAD_RETRY_DELAY_BASE: Final[Literal[2]] = 2

DEFAULT_TTL_SECONDS: Final[Literal[86_400]] = 86_400
DEFAULT_MAX_CACHE_ENTRIES: Final[Literal[1000]] = 1000

GEMINI_FILE_TTL: Final[timedelta] = timedelta(hours=48)
BACKOFF_BASE_DELAY: Final[float] = 1.0
BACKOFF_MAX_DELAY: Final[float] = 30.0
BACKOFF_JITTER_FACTOR: Final[float] = 0.1

FILES_API_MAX_SIZE: Final[Literal[536_870_912]] = 536_870_912
DEFAULT_UPLOAD_CHUNK_SIZE: Final[Literal[67_108_864]] = 67_108_864

MULTIPART_THRESHOLD: Final[Literal[8_388_608]] = 8_388_608
MULTIPART_CHUNKSIZE: Final[Literal[8_388_608]] = 8_388_608
MAX_CONCURRENCY: Final[Literal[10]] = 10
