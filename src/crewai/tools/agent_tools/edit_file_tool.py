from crewai.tools import BaseTool
from crewai.llm import LLM
from typing import Type, Optional
from pydantic import BaseModel, Field
from pathlib import Path


class EditFileToolInput(BaseModel):
    """Input schema for EditFileTool."""
    file_path: str = Field(..., description="Path to the file to edit")
    edit_instructions: str = Field(..., description="Clear instructions for what changes to make to the file")
    context: Optional[str] = Field(None, description="Additional context about the changes needed")


class EditFileTool(BaseTool):
    name: str = "edit_file"
    description: str = (
        "Edit files using Fast Apply model approach. Performs full-file rewrites instead of "
        "brittle search-and-replace operations. Provide clear edit instructions and the tool "
        "will generate an accurate, complete rewrite of the file with your changes applied."
    )
    args_schema: Type[BaseModel] = EditFileToolInput

    def __init__(self, llm: Optional[LLM] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLM(model="gpt-4o-mini", temperature=0.1)

    def _run(self, file_path: str, edit_instructions: str, context: Optional[str] = None) -> str:
        """
        Execute file editing using Fast Apply approach.
        
        Args:
            file_path: Path to the file to edit
            edit_instructions: Instructions for what changes to make
            context: Optional additional context
            
        Returns:
            Success message with details of the edit
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File {file_path} does not exist"
            
            if not path.is_file():
                return f"Error: {file_path} is not a file"
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    current_content = f.read()
            except UnicodeDecodeError:
                return f"Error: Cannot read {file_path} - file appears to be binary or uses unsupported encoding"
            
            prompt = self._build_fast_apply_prompt(
                current_content=current_content,
                edit_instructions=edit_instructions,
                file_path=file_path,
                context=context
            )
            
            response = self.llm.call(prompt)
            new_content = self._extract_file_content(response)
            
            if new_content is None:
                return "Error: Failed to generate valid file content. LLM response was malformed."
            
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(current_content)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            original_lines = len(current_content.splitlines())
            new_lines = len(new_content.splitlines())
            
            return (
                f"Successfully edited {file_path}. "
                f"Original: {original_lines} lines, New: {new_lines} lines. "
                f"Backup saved as {backup_path}"
            )
            
        except Exception as e:
            return f"Error editing file {file_path}: {str(e)}"

    def _build_fast_apply_prompt(self, current_content: str, edit_instructions: str, 
                                file_path: str, context: Optional[str] = None) -> str:
        """Build the Fast Apply prompt for the LLM."""
        file_extension = Path(file_path).suffix
        
        prompt = f"""You are an expert code editor implementing Fast Apply file editing. Your task is to rewrite the entire file with the requested changes applied.

IMPORTANT INSTRUCTIONS:
1. You must output the COMPLETE rewritten file content
2. Apply the edit instructions precisely while preserving all other functionality
3. Maintain the original file's style, formatting, and structure
4. Do not add explanatory comments unless they were in the original
5. Output ONLY the file content, no explanations or markdown formatting

FILE TO EDIT: {file_path}
FILE TYPE: {file_extension}

CURRENT FILE CONTENT:
```
{current_content}
```

EDIT INSTRUCTIONS:
{edit_instructions}"""

        if context:
            prompt += f"\n\nADDITIONAL CONTEXT:\n{context}"

        prompt += "\n\nOUTPUT THE COMPLETE REWRITTEN FILE CONTENT:"
        
        return prompt

    def _extract_file_content(self, llm_response: str) -> Optional[str]:
        """Extract the file content from LLM response."""
        content = llm_response.strip()
        
        if content.startswith('```') and content.endswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])
        
        return content if content else None
