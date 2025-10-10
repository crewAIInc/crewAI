from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List, Optional
import requests
import json
import re
import os
from datetime import datetime
import subprocess
import re

class KoreanFileMappingToolInput(BaseModel):
    """Input schema for Branch Change Analyzer Tool."""
    list_of_files: List[str] = Field(
        ..., 
        description="Files which have been changed in the feature branch"
    )

class KoreanFileMappingTool(BaseTool):
    """Tool for fetching comprehensive GitHub issue data using GitHub's REST API."""

    name: str = "Korean File Reverse Mapping"
    description: str = (
        "Find the files in korean directory which are related to the file which are changed"
        "Indexes the korean files and gives that back"
        "Input: list_of_files Output: A structure formatted response of what files and indexing of those files"
    )
    args_schema: Type[BaseModel] = KoreanFileMappingToolInput

    
    def index_markdown_lines(self,lines):
        index = []
        total_lines = len(lines)

        # Step 1: collect all headings with their start lines
        for i, line in enumerate(lines, start=1):
            if line.startswith("##") and not line.startswith("###"):
                # Count #s until first space
                level = line.count("#", 0, line.find(" ")) if " " in line else line.count("#")
                title = line.strip("# ").strip()
                index.append({
                    "level": level,
                    "title": title,
                    "start_line": i,
                    "end_line": None,
                })

        # Step 2: determine end_line for each heading
        for j in range(len(index)):
            current_level = index[j]["level"]
            current_start = index[j]["start_line"]

            # Find the next heading with same or higher level
            next_start = None
            for k in range(j + 1, len(index)):
                if index[k]["level"] <= current_level:
                    next_start = index[k]["start_line"]
                    break

            # If found, end is just before next heading
            end_line = (next_start - 1) if next_start else total_lines
            index[j]["end_line"] = end_line

        return index


    def _run(self, list_of_files: List[str]) -> dict:
        """Indentifies the korean files and returs the indexing of those files"""
        # Main execution
        reverse_section_mapping = {}
        for file in list_of_files:
            korean_file_path = file.replace("docs/en","docs/ko")
    
            parent_dir = os.path.dirname(os.getcwd())
            file_path = os.path.join(parent_dir, korean_file_path)
            
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            sections = self.index_markdown_lines(lines)
            reverse_section_mapping[korean_file_path] = sections

        
        return reverse_section_mapping