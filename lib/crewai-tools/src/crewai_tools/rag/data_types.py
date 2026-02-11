from enum import Enum
from importlib import import_module
import os
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from crewai_tools.rag.base_loader import BaseLoader
from crewai_tools.rag.chunkers.base_chunker import BaseChunker


class DataType(str, Enum):
    FILE = "file"
    PDF_FILE = "pdf_file"
    TEXT_FILE = "text_file"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    DOCX = "docx"
    MDX = "mdx"
    MYSQL = "mysql"
    POSTGRES = "postgres"
    GITHUB = "github"
    DIRECTORY = "directory"
    WEBSITE = "website"
    DOCS_SITE = "docs_site"
    YOUTUBE_VIDEO = "youtube_video"
    YOUTUBE_CHANNEL = "youtube_channel"
    TEXT = "text"

    def get_chunker(self) -> BaseChunker:
        from importlib import import_module

        chunkers = {
            DataType.PDF_FILE: ("text_chunker", "TextChunker"),
            DataType.TEXT_FILE: ("text_chunker", "TextChunker"),
            DataType.TEXT: ("text_chunker", "TextChunker"),
            DataType.DOCX: ("text_chunker", "DocxChunker"),
            DataType.MDX: ("text_chunker", "MdxChunker"),
            # Structured formats
            DataType.CSV: ("structured_chunker", "CsvChunker"),
            DataType.JSON: ("structured_chunker", "JsonChunker"),
            DataType.XML: ("structured_chunker", "XmlChunker"),
            DataType.WEBSITE: ("web_chunker", "WebsiteChunker"),
            DataType.DIRECTORY: ("text_chunker", "TextChunker"),
            DataType.YOUTUBE_VIDEO: ("text_chunker", "TextChunker"),
            DataType.YOUTUBE_CHANNEL: ("text_chunker", "TextChunker"),
            DataType.GITHUB: ("text_chunker", "TextChunker"),
            DataType.DOCS_SITE: ("text_chunker", "TextChunker"),
            DataType.MYSQL: ("text_chunker", "TextChunker"),
            DataType.POSTGRES: ("text_chunker", "TextChunker"),
        }

        if self not in chunkers:
            raise ValueError(f"No chunker defined for {self}")
        module_name, class_name = chunkers[self]
        module_path = f"crewai_tools.rag.chunkers.{module_name}"

        try:
            module = import_module(module_path)
            return cast(BaseChunker, getattr(module, class_name)())
        except Exception as e:
            raise ValueError(f"Error loading chunker for {self}: {e}") from e

    def get_loader(self) -> BaseLoader:
        loaders = {
            DataType.PDF_FILE: ("pdf_loader", "PDFLoader"),
            DataType.TEXT_FILE: ("text_loader", "TextFileLoader"),
            DataType.TEXT: ("text_loader", "TextLoader"),
            DataType.XML: ("xml_loader", "XMLLoader"),
            DataType.WEBSITE: ("webpage_loader", "WebPageLoader"),
            DataType.MDX: ("mdx_loader", "MDXLoader"),
            DataType.JSON: ("json_loader", "JSONLoader"),
            DataType.DOCX: ("docx_loader", "DOCXLoader"),
            DataType.CSV: ("csv_loader", "CSVLoader"),
            DataType.DIRECTORY: ("directory_loader", "DirectoryLoader"),
            DataType.YOUTUBE_VIDEO: ("youtube_video_loader", "YoutubeVideoLoader"),
            DataType.YOUTUBE_CHANNEL: (
                "youtube_channel_loader",
                "YoutubeChannelLoader",
            ),
            DataType.GITHUB: ("github_loader", "GithubLoader"),
            DataType.DOCS_SITE: ("docs_site_loader", "DocsSiteLoader"),
            DataType.MYSQL: ("mysql_loader", "MySQLLoader"),
            DataType.POSTGRES: ("postgres_loader", "PostgresLoader"),
        }

        if self not in loaders:
            raise ValueError(f"No loader defined for {self}")
        module_name, class_name = loaders[self]
        module_path = f"crewai_tools.rag.loaders.{module_name}"
        try:
            module = import_module(module_path)
            return cast(BaseLoader, getattr(module, class_name)())
        except Exception as e:
            raise ValueError(f"Error loading loader for {self}: {e}") from e


class DataTypes:
    @staticmethod
    def from_content(content: str | Path | None = None) -> DataType:
        if content is None:
            return DataType.TEXT

        if isinstance(content, Path):
            content = str(content)

        is_url = False
        if isinstance(content, str):
            try:
                url = urlparse(content)
                is_url = bool(url.scheme and url.netloc) or url.scheme == "file"
            except Exception:  # noqa: S110
                pass

        def get_file_type(path: str) -> DataType | None:
            mapping = {
                ".pdf": DataType.PDF_FILE,
                ".csv": DataType.CSV,
                ".mdx": DataType.MDX,
                ".md": DataType.MDX,
                ".docx": DataType.DOCX,
                ".json": DataType.JSON,
                ".xml": DataType.XML,
                ".txt": DataType.TEXT_FILE,
            }
            for ext, dtype in mapping.items():
                if path.endswith(ext):
                    return dtype
            return None

        if is_url:
            dtype = get_file_type(url.path)
            if dtype:
                return dtype

            if "docs" in url.netloc or ("docs" in url.path and url.scheme != "file"):
                return DataType.DOCS_SITE
            if "github.com" in url.netloc:
                return DataType.GITHUB

            return DataType.WEBSITE

        if os.path.isfile(content):
            dtype = get_file_type(content)
            if dtype:
                return dtype

            if os.path.exists(content):
                return DataType.TEXT_FILE
        elif os.path.isdir(content):
            return DataType.DIRECTORY

        return DataType.TEXT
