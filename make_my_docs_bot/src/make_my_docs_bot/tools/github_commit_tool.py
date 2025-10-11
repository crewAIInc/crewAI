from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List
import subprocess
import os


class GitCommitToolInput(BaseModel):
    """Input schema for Git Commit Tool."""
    feature_branch: str = Field(..., description="The feature branch to checkout")
    files_to_commit: List[str] = Field(..., description="List of file paths to add and commit")
    commit_message: str = Field(..., description="Commit message for the changes")


class GitCommitTool(BaseTool):
    """Tool for committing changes to a Git feature branch."""

    name: str = "Git Commit Tool"
    description: str = (
        "Checks out the given feature branch, adds the specified files, "
        "and commits them with the provided commit message."
    )
    args_schema: Type[BaseModel] = GitCommitToolInput

    def _run(
        self,
        feature_branch: str,
        files_to_commit: List[str],
        commit_message: str
    ) -> str:
        """Perform git checkout, add files, and commit."""
        try:
            # Checkout the branch
            subprocess.run(["git", "checkout", feature_branch], check=True)

            # Add files
            if not files_to_commit:
                return {"error": "No files provided to commit."}
            
            parent_dir = os.path.dirname(os.getcwd())

            for index in range(len(files_to_commit)):
                files_to_commit[index] = os.path.join(parent_dir, files_to_commit[index])
            
            # Add files
            subprocess.run(["git", "add"] + files_to_commit, check=True)
            print("✅ Files successfully added to staging area.")

            # Commit changes
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            print(f"✅ Successfully committed with message: '{commit_message}'")

            return {"message": f"Changes committed to branch '{feature_branch}' successfully."}

        except subprocess.CalledProcessError as e:
            return {"error": f"Git command failed: {e}"}
