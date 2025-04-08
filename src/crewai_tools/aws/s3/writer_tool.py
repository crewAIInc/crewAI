from typing import Any, Type
import os

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class S3WriterToolInput(BaseModel):
    """Input schema for S3WriterTool."""
    file_path: str = Field(..., description="S3 file path (e.g., 's3://bucket-name/file-name')")
    content: str = Field(..., description="Content to write to the file")


class S3WriterTool(BaseTool):
    name: str = "S3 Writer Tool"
    description: str = "Writes content to a file in Amazon S3 given an S3 file path"
    args_schema: Type[BaseModel] = S3WriterToolInput

    def _run(self, file_path: str, content: str) -> str:
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("`boto3` package not found, please run `uv add boto3`")

        try:
            bucket_name, object_key = self._parse_s3_path(file_path)

            s3 = boto3.client(
                's3',
                region_name=os.getenv('CREW_AWS_REGION', 'us-east-1'),
                aws_access_key_id=os.getenv('CREW_AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('CREW_AWS_SEC_ACCESS_KEY')
            )

            s3.put_object(Bucket=bucket_name, Key=object_key, Body=content.encode('utf-8'))
            return f"Successfully wrote content to {file_path}"
        except ClientError as e:
            return f"Error writing file to S3: {str(e)}"

    def _parse_s3_path(self, file_path: str) -> tuple:
        parts = file_path.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]
