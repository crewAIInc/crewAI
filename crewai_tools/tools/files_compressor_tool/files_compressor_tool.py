import os
import zipfile
import tarfile
from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class FileCompressorToolInput(BaseModel):
    """Input schema for FileCompressorTool."""
    input_path: str = Field(..., description="Path to the file or directory to compress.")
    output_path: Optional[str] = Field(default=None, description="Optional output archive filename.")
    overwrite: bool = Field(default=False, description="Whether to overwrite the archive if it already exists.")
    format: str = Field(default="zip", description="Compression format ('zip', 'tar', 'tar.gz', 'tar.bz2', 'tar.xz').")


class FileCompressorTool(BaseTool):
    name: str = "File Compressor Tool"
    description: str = (
        "Compresses a file or directory into an archive (.zip currently supported). "
        "Useful for archiving logs, documents, or backups."
    )
    args_schema: Type[BaseModel] = FileCompressorToolInput

    
    def _run(self, input_path: str, output_path: Optional[str] = None, overwrite: bool = False, format: str = "zip") -> str:
            
            if not os.path.exists(input_path):
                return f"Input path '{input_path}' does not exist."
            
            if not output_path:
                output_path = self._generate_output_path(input_path, format)
            
            FORMAT_EXTENSION = {
                    "zip": ".zip",
                    "tar": ".tar",
                    "tar.gz": ".tar.gz",
                    "tar.bz2": ".tar.bz2",
                    "tar.xz": ".tar.xz"
                }
            
            if format not in FORMAT_EXTENSION:
                return f"Compression format '{format}' is not supported. Allowed formats: {', '.join(FORMAT_EXTENSION.keys())}"
            elif not output_path.endswith(FORMAT_EXTENSION[format]):
                return f"Error: If '{format}' format is chosen, output file must have a '{FORMAT_EXTENSION[format]}' extension."
            if not self._prepare_output(output_path, overwrite):
                return f"Output '{output_path}' already exists and overwrite is set to False."

            try:
                format_compression = {
                    "zip": self._compress_zip,
                    "tar": self._compress_tar,
                    "tar.gz": self._compress_tar,
                    "tar.bz2": self._compress_tar,
                    "tar.xz": self._compress_tar
                }
                if format == "zip":
                    format_compression[format](input_path, output_path) 
                else:
                    format_compression[format](input_path, output_path, format)
                
                return f"Successfully compressed '{input_path}' into '{output_path}'"
            except FileNotFoundError:
                return f"Error: File not found at path: {input_path}"
            except PermissionError:
                return f"Error: Permission denied when accessing '{input_path}' or writing '{output_path}'"
            except Exception as e:
                return f"An unexpected error occurred during compression: {str(e)}"


    def _generate_output_path(self, input_path: str, format: str) -> str:
        """Generates output path based on input path and format."""
        if os.path.isfile(input_path):
            base_name = os.path.splitext(os.path.basename(input_path))[0]  # Remove extension
        else:
            base_name = os.path.basename(os.path.normpath(input_path))  # Directory name
        return os.path.join(os.getcwd(), f"{base_name}.{format}")

    def _prepare_output(self, output_path: str, overwrite: bool) -> bool:
        """Ensures output path is ready for writing."""
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.path.exists(output_path) and not overwrite:
            return False
        return True

    def _compress_zip(self, input_path: str, output_path: str):
        """Compresses input into a zip archive."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.isfile(input_path):
                zipf.write(input_path, os.path.basename(input_path))
            else:
                for root, _, files in os.walk(input_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, start=input_path)
                        zipf.write(full_path, arcname)


    def _compress_tar(self, input_path: str, output_path: str, format: str):
        """Compresses input into a tar archive with the given format."""
        format_mode = {
            "tar": "w",
            "tar.gz": "w:gz",
            "tar.bz2": "w:bz2",
            "tar.xz": "w:xz"
        }

        if format not in format_mode:
            raise ValueError(f"Unsupported tar format: {format}")
        
        mode = format_mode[format]

        with tarfile.open(output_path, mode) as tarf:
            arcname = os.path.basename(input_path)
            tarf.add(input_path, arcname=arcname)
