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

class BranchChangeAnalyzerToolInput(BaseModel):
    """Input schema for Branch Change Analyzer Tool."""
    feature_branch: str = Field(
        ..., 
        description="Branch name against which diff will be calculated"
    )
    file_path: str = Field(
        ..., 
        description="File path in which changes needs to be identified"
    )

class BranchChangeAnalyzerTool(BaseTool):
    """Tool for fetching comprehensive GitHub issue data using GitHub's REST API."""

    name: str = "Branch Change Analyzer Tool"
    description: str = (
        "Fetches the changed files between the feature branch and the main branch of the given file_path file"
        "Indexes the changed files and find outs the changed section in a structed format"
        "Input: feature_branch Output: A structure formatted response of what files and lines changed"
    )
    args_schema: Type[BaseModel] = BranchChangeAnalyzerToolInput

    
    def parse_git_diff(self,file_path, base_branch, compare_branch):
        # Get parent directory of current working directory
        parent_dir = os.path.dirname(os.getcwd())
        cmd = ["git", "diff", "-U0", f"{base_branch}...{compare_branch}", "--", file_path]
        result = subprocess.run(cmd, cwd=parent_dir, capture_output=True, text=True)
        diff_text = result.stdout

        changed_lines = []
        for hunk in re.finditer(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff_text):
            start = int(hunk.group(1))
            length = int(hunk.group(2) or 1)
            changed_lines.extend(range(start, start + length))

        return changed_lines
    

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
                    "content": None
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

            # Extract content between start_line and end_line (exclusive of heading)
            content_lines = lines[current_start:end_line]
            index[j]["content"] = "".join(content_lines).strip()

        return index


    def find_section_for_line(self,line_no, sections):
        prev = None
        for sec in sections:
            if (sec["start_line"] > line_no) and (sec["end_line"] > line_no):
                break
            prev = sec
        return prev["title"] if prev else None


    def _run(self, feature_branch: str, file_path: str) -> str:
        """Fetch and return GitHub issue data."""
        # Main execution

        changed_files = [file_path]
        if not changed_files:
            return {"message": "No .mdx files changed in docs/en/ directory"}
        
        changed_section_metadata_parent = {}
        for changed_file in changed_files:
            changed_lines = self.parse_git_diff(changed_file,"main",feature_branch) ## This will typically give me the lines which have changed

            parent_dir = os.path.dirname(os.getcwd())
            file_path = os.path.join(parent_dir, changed_file)
            
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            sections = self.index_markdown_lines(lines)
            changed_sections = []
            for line in changed_lines:
                section = self.find_section_for_line(line,sections)

                if (len(changed_sections) == 0) or (len(changed_sections) and changed_sections[-1] != section):
                    changed_sections.append(section)
            
            changed_section_metadata = []
            for changed_section in changed_sections:
                for section in sections:
                    if section['title'] == changed_section:
                        changed_section_metadata.append({'title': section['title'], 'content': section['content']})
            
            if len(changed_section_metadata):
                changed_section_metadata.reverse()
                changed_section_metadata_parent[changed_file] = changed_section_metadata
        
        return {
            "CHANGED_SECTION_PER_FILE" : changed_section_metadata_parent,
            "CHANGED_FILES" : list(changed_section_metadata_parent.keys())
        }