"""Unit tests for file_store module."""

import uuid

import pytest

from crewai.utilities.file_store import (
    clear_files,
    clear_task_files,
    get_all_files,
    get_files,
    get_task_files,
    store_files,
    store_task_files,
)
from crewai_files import TextFile


class TestFileStore:
    """Tests for synchronous file store operations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.crew_id = uuid.uuid4()
        self.task_id = uuid.uuid4()
        self.test_file = TextFile(source=b"test content")

    def teardown_method(self) -> None:
        """Clean up after tests."""
        clear_files(self.crew_id)
        clear_task_files(self.task_id)

    def test_store_and_get_files(self) -> None:
        """Test storing and retrieving crew files."""
        files = {"doc": self.test_file}
        store_files(self.crew_id, files)

        retrieved = get_files(self.crew_id)

        assert retrieved is not None
        assert "doc" in retrieved
        assert retrieved["doc"].read() == b"test content"

    def test_get_files_returns_none_when_empty(self) -> None:
        """Test that get_files returns None for non-existent keys."""
        new_id = uuid.uuid4()
        result = get_files(new_id)
        assert result is None

    def test_clear_files(self) -> None:
        """Test clearing crew files."""
        files = {"doc": self.test_file}
        store_files(self.crew_id, files)

        clear_files(self.crew_id)

        result = get_files(self.crew_id)
        assert result is None

    def test_store_and_get_task_files(self) -> None:
        """Test storing and retrieving task files."""
        files = {"task_doc": self.test_file}
        store_task_files(self.task_id, files)

        retrieved = get_task_files(self.task_id)

        assert retrieved is not None
        assert "task_doc" in retrieved

    def test_clear_task_files(self) -> None:
        """Test clearing task files."""
        files = {"task_doc": self.test_file}
        store_task_files(self.task_id, files)

        clear_task_files(self.task_id)

        result = get_task_files(self.task_id)
        assert result is None

    def test_get_all_files_merges_crew_and_task(self) -> None:
        """Test that get_all_files merges crew and task files."""
        crew_file = TextFile(source=b"crew content")
        task_file = TextFile(source=b"task content")

        store_files(self.crew_id, {"crew_doc": crew_file})
        store_task_files(self.task_id, {"task_doc": task_file})

        merged = get_all_files(self.crew_id, self.task_id)

        assert merged is not None
        assert "crew_doc" in merged
        assert "task_doc" in merged

    def test_get_all_files_task_overrides_crew(self) -> None:
        """Test that task files override crew files with same name."""
        crew_file = TextFile(source=b"crew version")
        task_file = TextFile(source=b"task version")

        store_files(self.crew_id, {"shared_doc": crew_file})
        store_task_files(self.task_id, {"shared_doc": task_file})

        merged = get_all_files(self.crew_id, self.task_id)

        assert merged is not None
        assert merged["shared_doc"].read() == b"task version"

    def test_get_all_files_crew_only(self) -> None:
        """Test get_all_files with only crew files."""
        store_files(self.crew_id, {"doc": self.test_file})

        result = get_all_files(self.crew_id)

        assert result is not None
        assert "doc" in result

    def test_get_all_files_returns_none_when_empty(self) -> None:
        """Test that get_all_files returns None when no files exist."""
        new_crew_id = uuid.uuid4()
        new_task_id = uuid.uuid4()

        result = get_all_files(new_crew_id, new_task_id)

        assert result is None


@pytest.mark.asyncio
class TestAsyncFileStore:
    """Tests for asynchronous file store operations."""

    async def test_astore_and_aget_files(self) -> None:
        """Test async storing and retrieving crew files."""
        from crewai.utilities.file_store import aclear_files, aget_files, astore_files

        crew_id = uuid.uuid4()
        test_file = TextFile(source=b"async content")

        try:
            await astore_files(crew_id, {"doc": test_file})
            retrieved = await aget_files(crew_id)

            assert retrieved is not None
            assert "doc" in retrieved
            assert retrieved["doc"].read() == b"async content"
        finally:
            await aclear_files(crew_id)

    async def test_aget_all_files(self) -> None:
        """Test async get_all_files merging."""
        from crewai.utilities.file_store import (
            aclear_files,
            aclear_task_files,
            aget_all_files,
            astore_files,
            astore_task_files,
        )

        crew_id = uuid.uuid4()
        task_id = uuid.uuid4()

        try:
            await astore_files(crew_id, {"crew": TextFile(source=b"crew")})
            await astore_task_files(task_id, {"task": TextFile(source=b"task")})

            merged = await aget_all_files(crew_id, task_id)

            assert merged is not None
            assert "crew" in merged
            assert "task" in merged
        finally:
            await aclear_files(crew_id)
            await aclear_task_files(task_id)