from collections.abc import Iterator
from contextlib import contextmanager
import errno
import os
import tarfile
from typing import BinaryIO
import zipfile

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crewai_tools.security.safe_path import validate_file_path


_SUPPORTS_SECURE_DIR_FD = (
    os.open in os.supports_dir_fd
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)


class FileCompressorToolInput(BaseModel):
    """Input schema for FileCompressorTool."""

    input_path: str = Field(
        ..., description="Path to the file or directory to compress."
    )
    output_path: str | None = Field(
        default=None, description="Optional output archive filename."
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite the archive if it already exists.",
    )
    format: str = Field(
        default="zip",
        description="Compression format ('zip', 'tar', 'tar.gz', 'tar.bz2', 'tar.xz').",
    )


class FileCompressorTool(BaseTool):
    name: str = "File Compressor Tool"
    description: str = (
        "Compresses a file or directory into an archive (.zip currently supported). "
        "Useful for archiving logs, documents, or backups."
    )
    args_schema: type[BaseModel] = FileCompressorToolInput

    def _run(
        self,
        input_path: str,
        output_path: str | None = None,
        overwrite: bool = False,
        format: str = "zip",
    ) -> str:
        """Compress an input path while protecting source files from output overlap."""
        input_path = validate_file_path(input_path)
        if not os.path.exists(input_path):
            return f"Input path '{input_path}' does not exist."

        if not output_path:
            output_path = self._generate_output_path(input_path, format)

        # Keep the caller's spelling: ``validate_file_path`` resolves symlinks, and the overlap
        # guard has to know which path the caller actually named as the output — otherwise an
        # output symlink pointing into the input tree looks like the in-tree file itself.
        requested_output_path = output_path
        output_path = validate_file_path(output_path)

        format_extension = {
            "zip": ".zip",
            "tar": ".tar",
            "tar.gz": ".tar.gz",
            "tar.bz2": ".tar.bz2",
            "tar.xz": ".tar.xz",
        }

        if format not in format_extension:
            return f"Compression format '{format}' is not supported. Allowed formats: {', '.join(format_extension.keys())}"
        if not output_path.endswith(format_extension[format]):
            return f"Error: If '{format}' format is chosen, output file must have a '{format_extension[format]}' extension."
        if not self._prepare_output(output_path, overwrite):
            return (
                f"Output '{output_path}' already exists and overwrite is set to False."
            )

        try:
            format_compression = {
                "zip": self._compress_zip,
                "tar": self._compress_tar,
                "tar.gz": self._compress_tar,
                "tar.bz2": self._compress_tar,
                "tar.xz": self._compress_tar,
            }
            if format == "zip":
                format_compression[format](  # type: ignore[operator]
                    input_path,
                    output_path,
                    requested_output_path,
                    overwrite,
                )
            else:
                format_compression[format](  # type: ignore[operator]
                    input_path,
                    output_path,
                    requested_output_path,
                    overwrite,
                    format,
                )

            return f"Successfully compressed '{input_path}' into '{output_path}'"
        except FileNotFoundError:
            return f"Error: File not found at path: {input_path}"
        except PermissionError:
            return f"Error: Permission denied when accessing '{input_path}' or writing '{output_path}'"
        except FileExistsError:
            return (
                f"Output '{output_path}' already exists and overwrite is set to False."
            )
        except ValueError as e:
            # A rejected input/output overlap is a usage error, not an unexpected failure.
            return f"Error: {e!s}"
        except Exception as e:
            return f"An unexpected error occurred during compression: {e!s}"

    @staticmethod
    def _generate_output_path(input_path: str, format: str) -> str:
        """Generates output path based on input path and format."""
        if os.path.isfile(input_path):
            base_name = os.path.splitext(os.path.basename(input_path))[0]
        else:
            base_name = os.path.basename(os.path.normpath(input_path))
        return os.path.join(os.getcwd(), f"{base_name}.{format}")

    @staticmethod
    def _prepare_output(output_path: str, overwrite: bool) -> bool:
        """Ensures output path is ready for writing."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_path) and not overwrite:
            return False
        return True

    @staticmethod
    def _reject_output_aliasing_input(
        input_path: str,
        output_path: str,
        requested_output_path: str,
        output_fd: int,
    ) -> None:
        """Raise when the opened output IS a file being compressed.

        The output is already open without truncation. Comparing its descriptor to each input's
        filesystem identity closes the check-to-open race: changing ``output_path`` now cannot
        redirect the later archive write to an input file.

        Only an alias of an *existing* input file is rejected. A new output inside the tree is the
        normal case and is excluded from the archive instead (see ``_compress_zip``).

        ``requested_output_path`` is the caller's spelling, before ``validate_file_path`` resolved
        it. The exemption below has to use that: ``output_path`` is already canonical, so an output
        symlink pointing into the tree would otherwise look like the in-tree file itself.
        """
        # Resolve parent aliases to match the canonical input walk, but preserve the final name.
        # Resolving that last component would exempt a symlink's target and allow source loss.
        requested_parent, requested_name = os.path.split(requested_output_path)
        requested_abs_path = os.path.normcase(
            os.path.join(os.path.realpath(requested_parent), requested_name)
        )
        output_stat = os.fstat(output_fd)

        def _aliases(candidate: str) -> bool:
            try:
                return os.path.samestat(os.stat(candidate), output_stat)
            except OSError:
                return False

        if os.path.isfile(input_path):
            if _aliases(input_path):
                raise ValueError(
                    f"Input and output are the same file: '{input_path}'. Compressing it would "
                    f"truncate the source."
                )
            return

        for root, _, files in os.walk(input_path):
            for name in files:
                candidate = os.path.join(root, name)
                if os.path.normcase(os.path.abspath(candidate)) == requested_abs_path:
                    # Overwriting the archive the caller named is an ordinary overwrite.
                    continue
                if _aliases(candidate):
                    raise ValueError(
                        f"Output '{output_path}' is the same file as '{candidate}' inside the "
                        f"input. Compressing it would truncate that file; use a different output."
                    )

    @staticmethod
    def _open_output_fd(output_path: str, flags: int) -> int:
        """Open an output without following a replaced path component.

        On platforms with descriptor-relative opens, every directory component is pinned before
        the next component is opened. This prevents a concurrent rename or symlink substitution
        from redirecting the final open to a different writable file.
        """
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        flags |= nofollow

        if not _SUPPORTS_SECURE_DIR_FD or not os.path.isabs(output_path):
            return os.open(output_path, flags, 0o666)

        components = [part for part in output_path.split(os.sep) if part]
        directory_flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | nofollow
        directory_fd = os.open(os.sep, directory_flags)
        try:
            for component in components[:-1]:
                next_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd

            return os.open(
                components[-1],
                flags,
                0o666,
                dir_fd=directory_fd,
            )
        finally:
            os.close(directory_fd)

    @staticmethod
    @contextmanager
    def _open_validated_output(
        input_path: str,
        output_path: str,
        requested_output_path: str,
        overwrite: bool,
    ) -> Iterator[BinaryIO]:
        """Open, validate, and truncate one descriptor for the complete archive write."""
        flags = os.O_RDWR | os.O_CREAT
        if not overwrite:
            flags |= os.O_EXCL

        try:
            output_fd = FileCompressorTool._open_output_fd(output_path, flags)
        except OSError as error:
            unsafe_path_errors = {errno.ELOOP, errno.ENOTDIR}
            if error.errno in unsafe_path_errors:
                raise ValueError(
                    f"Output '{output_path}' changed to an unsafe symbolic-link path before it "
                    "could be opened."
                ) from error
            raise
        with os.fdopen(output_fd, "r+b") as output_file:
            FileCompressorTool._reject_output_aliasing_input(
                input_path,
                output_path,
                requested_output_path,
                output_file.fileno(),
            )
            output_file.seek(0)
            output_file.truncate(0)
            yield output_file

    @staticmethod
    def _compress_zip(
        input_path: str,
        output_path: str,
        requested_output_path: str,
        overwrite: bool,
    ) -> None:
        """Compresses input into a zip archive."""
        with FileCompressorTool._open_validated_output(
            input_path, output_path, requested_output_path, overwrite
        ) as output_file:
            output_stat = os.fstat(output_file.fileno())
            with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                if os.path.isfile(input_path):
                    zipf.write(input_path, os.path.basename(input_path))
                else:
                    for root, _, files in os.walk(input_path):
                        for file in files:
                            full_path = os.path.join(root, file)
                            try:
                                if os.path.samestat(os.stat(full_path), output_stat):
                                    continue
                            except OSError:
                                pass
                            arcname = os.path.relpath(full_path, start=input_path)
                            zipf.write(full_path, arcname)

    @staticmethod
    def _compress_tar(
        input_path: str,
        output_path: str,
        requested_output_path: str,
        overwrite: bool,
        format: str,
    ) -> None:
        """Compresses input into a tar archive with the given format."""
        format_mode = {
            "tar": "w",
            "tar.gz": "w:gz",
            "tar.bz2": "w:bz2",
            "tar.xz": "w:xz",
        }

        if format not in format_mode:
            raise ValueError(f"Unsupported tar format: {format}")

        mode = format_mode[format]

        with FileCompressorTool._open_validated_output(
            input_path, output_path, requested_output_path, overwrite
        ) as output_file:
            with tarfile.open(output_path, mode, fileobj=output_file) as tarf:  # type: ignore[call-overload]
                arcname = os.path.basename(input_path)
                tarf.add(input_path, arcname=arcname)
