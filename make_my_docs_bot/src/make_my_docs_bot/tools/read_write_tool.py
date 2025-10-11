from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import os


class FileContentUpdaterToolInput(BaseModel):
    """Input schema for File Content Updater Tool."""
    file_path: str = Field(..., description="Path of the file to be modified")
    content_to_be_deleted_start_line: int = Field(..., description="Start line number (1-based) from which data will be erased")
    content_to_be_deleted_end_line: int = Field(..., description="End line number (1-based) till which data will be erased")
    new_content: str = Field(..., description="New content that needs to be added in place of deleted lines")


class FileContentUpdaterTool(BaseTool):
    """Tool for replacing a specific line range in a file with new content."""

    name: str = "File Content Updater Tool"
    description: str = (
        "Opens a file and replaces lines between start and end line numbers "
        "with the provided new content. Overwrites the file with the updated text."
    )
    args_schema: Type[BaseModel] = FileContentUpdaterToolInput

    def _run(
        self,
        file_path: str,
        content_to_be_deleted_start_line: int,
        content_to_be_deleted_end_line: int,
        new_content: str
    ) -> str:
        """Replace lines in a file between given start and end line numbers."""
        # Read file
        parent_dir = os.path.dirname(os.getcwd())
        file_path = os.path.join(parent_dir, file_path)

        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start = content_to_be_deleted_start_line
        end = content_to_be_deleted_end_line

        # Validate line range
        if start < 1 or end > total_lines or start > end:
            return {
                "error": f"Invalid line range: file has {total_lines} lines, got start={start}, end={end}"
            }

        # Prepare new content (ensure newline handling)
        new_lines = [line + "\n" if not line.endswith("\n") else line for line in new_content.splitlines()]

        # Replace the specified range
        updated_lines = lines[: start - 1] + new_lines + lines[end:]

        # Write updated file
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

        print(f"✅ Successfully updated file: {file_path}")

        return {
            "message": f"Lines {start}-{end} in '{file_path}' replaced successfully.",
            "new_content_preview": new_content[:300] + ("..." if len(new_content) > 300 else "")
        }
