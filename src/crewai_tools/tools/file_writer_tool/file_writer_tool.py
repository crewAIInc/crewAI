import os
from crewai_tools import BaseTool

class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    description: str = "A tool to write content to a specified file. Accepts filename, content, and optionally a directory path as input."

    def _run(self, filename: str, content: str, directory: str = '.') -> str:
        try:
            # Create the directory if it doesn't exist
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Construct the full path
            filepath = os.path.join(directory, filename)

            # Write content to the file
            with open(filepath, 'w') as file:
                file.write(content)
            return f"Content successfully written to {filepath}"
        except Exception as e:
            return f"An error occurred while writing to the file: {str(e)}"
