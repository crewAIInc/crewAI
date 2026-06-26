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
                skipped = format_compression[format](input_path, output_path)  # type: ignore[operator]
            else:
                skipped = format_compression[format](  # type: ignore[operator]
                    input_path, output_path, format
                )

            message = f"Successfully compressed '{input_path}' into '{output_path}'"
            if isinstance(skipped, list) and skipped:
                message += (
                    f" ({len(skipped)} item(s) skipped for safety: resolved "
                    f"outside the allowed directory)"
                )
            return message
        except FileNotFoundError:
            return f"Error: File not found at path: {input_path}"
        except PermissionError:
            return f"Error: Permission denied when accessing '{input_path}' or writing '{output_path}'"
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
    def _compress_zip(input_path: str, output_path: str) -> list[str]:
        """Compresses input into a zip archive.

        Returns the files skipped because they resolved outside the allow-list.
        ``zipfile.write`` dereferences symlinks, so each walked file is
        validated to stop a symlink inside the tree from copying the contents
        of an out-of-tree target (e.g. ``~/.ssh/id_rsa``) into the archive.
        """
        # Defense in depth: validate the write target at the sink, so this is
        # safe even if called directly rather than through _run.
        output_path = validate_file_path(output_path)
        input_path = validate_file_path(input_path)
        skipped: list[str] = []
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if os.path.isfile(input_path):
                zipf.write(input_path, os.path.basename(input_path))
            else:
                for root, _, files in os.walk(input_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        try:
                            resolved_path = validate_file_path(full_path)
                        except ValueError:
                            skipped.append(full_path)
                            continue
                        arcname = os.path.relpath(full_path, start=input_path)
                        zipf.write(resolved_path, arcname)
        return skipped

    @staticmethod
    def _compress_tar(input_path: str, output_path: str, format: str) -> list[str]:
        """Compresses input into a tar archive with the given format.

        Returns the members skipped for safety. ``tarfile`` stores symlinks and
        hardlinks as link entries rather than dereferencing them, so the
        compress-time content leak that affects zip does not apply here. The
        filter drops link members so an out-of-tree symlink target cannot be
        shipped inside the archive and resolved at extraction time.
        """
        format_mode = {
            "tar": "w",
            "tar.gz": "w:gz",
            "tar.bz2": "w:bz2",
            "tar.xz": "w:xz",
        }

        if format not in format_mode:
            raise ValueError(f"Unsupported tar format: {format}")

        mode = format_mode[format]
        skipped: list[str] = []

        def _drop_links(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
            if tarinfo.issym() or tarinfo.islnk():
                skipped.append(tarinfo.name)
                return None
            return tarinfo

        # Defense in depth: validate the write target at the sink, so this is
        # safe even if called directly rather than through _run.
        output_path = validate_file_path(output_path)
        input_path = validate_file_path(input_path)
        with tarfile.open(output_path, mode) as tarf:  # type: ignore[call-overload]
            arcname = os.path.basename(input_path)
            tarf.add(input_path, arcname=arcname, filter=_drop_links)
        return skipped
