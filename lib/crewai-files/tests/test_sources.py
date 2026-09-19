"""Tests for file source constructors."""

from tempfile import TemporaryFile

from crewai_files import FileStream


def test_filestream_accepts_integer_descriptor_name():
    # Regression for #7621: streams opened from a file descriptor expose an
    # integer ``name``. Filename inference must skip Path() in that case so
    # the optional filename stays unset and the stream remains readable.
    with TemporaryFile() as original:
        original.write(b"A file supplied to an agent.")
        original.seek(0)
        with open(original.fileno(), "rb", closefd=False) as stream:
            source = FileStream(stream=stream)
            assert source.filename is None
            assert source.read() == b"A file supplied to an agent."


def test_filestream_infers_filename_from_named_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"hello")
    with path.open("rb") as stream:
        source = FileStream(stream=stream)
        assert source.filename == "notes.txt"
        assert source.read() == b"hello"


def test_filestream_keeps_explicit_filename():
    with TemporaryFile() as original:
        original.write(b"data")
        original.seek(0)
        with open(original.fileno(), "rb", closefd=False) as stream:
            source = FileStream(stream=stream, filename="agent.bin")
            assert source.filename == "agent.bin"
            assert source.read() == b"data"
