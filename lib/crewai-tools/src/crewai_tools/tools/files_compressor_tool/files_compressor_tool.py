import os
import tarfile
import zipfile

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crewai_tools.security.safe_path import validate_file_path


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
        input_path = validate_file_path(input_path)
        if not os.path.exists(input_path):
            return f"Input path '{input_path}' does not exist."

        if not output_path:
            output_path = self._generate_output_path(input_path, format)

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
                format_compression[format](input_path, output_path)  # type: ignore[operator]
            else:
                format_compression[format](input_path, output_path, format)  # type: ignore[operator]

            return f"Successfully compressed '{input_path}' into '{output_path}'"
        except FileNotFoundError:
            return f"Error: File not found at path: {input_path}"
        except PermissionError:
            return f"Error: Permission denied when accessing '{input_path}' or writing '{output_path}'"
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
    def _reject_output_aliasing_input(input_path: str, output_path: str) -> None:
        """Raise when ``output_path`` IS a file being compressed.

        ``ZipFile(output_path, "w")`` truncates its target before anything is read, so an output
        that aliases an input file destroys it before the walk could skip it. Comparing paths is not
        enough: a hard link has its own path but shares the inode, so this compares filesystem
        identity over the input tree — before the archive is opened.

        Only an alias of an *existing* input file is rejected. A new output inside the tree is the
        normal case and is excluded from the archive instead (see ``_compress_zip``).
        """
        if not os.path.exists(output_path):
            return
        # Exempt the output's own path only — lexically, not by ``realpath``. A symlink in the tree
        # that resolves to the output has a different path but the same target, so exempting
        # everything that resolves there would let opening the output truncate it.
        output_abs_path = os.path.abspath(output_path)

        def _aliases(candidate: str) -> bool:
            try:
                return os.path.samefile(candidate, output_path)
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
                if os.path.abspath(candidate) == output_abs_path:
                    # Overwriting the archive that is already there is an ordinary overwrite.
                    continue
                if _aliases(candidate):
                    raise ValueError(
                        f"Output '{output_path}' is the same file as '{candidate}' inside the "
                        f"input. Compressing it would truncate that file; use a different output."
                    )

    @staticmethod
    def _compress_zip(input_path: str, output_path: str) -> None:
        """Compresses input into a zip archive."""
        # Both the self-inclusion guard and the truncation guard have to settle before the archive
        # is opened: opening it creates (or truncates) the output.
        FileCompressorTool._reject_output_aliasing_input(input_path, output_path)
        # Opening the archive creates it, so when it lands inside ``input_path`` the walk below
        # would otherwise add the archive to itself: an empty, self-referential member that was
        # never in the source directory. ``tarfile`` guards against this internally; ``zipfile``
        # does not, so resolve the output once and skip that entry.
        output_real_path = os.path.realpath(output_path)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if os.path.isfile(input_path):
                zipf.write(input_path, os.path.basename(input_path))
            else:
                for root, _, files in os.walk(input_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        if os.path.realpath(full_path) == output_real_path:
                            continue
                        arcname = os.path.relpath(full_path, start=input_path)
                        zipf.write(full_path, arcname)

    @staticmethod
    def _compress_tar(input_path: str, output_path: str, format: str) -> None:
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

        with tarfile.open(output_path, mode) as tarf:  # type: ignore[call-overload]
            arcname = os.path.basename(input_path)
            tarf.add(input_path, arcname=arcname)
