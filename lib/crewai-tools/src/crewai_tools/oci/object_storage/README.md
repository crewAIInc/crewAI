# OCI Object Storage Tools

These tools let CrewAI agents read from and write to Oracle Cloud Infrastructure Object Storage.

## Installation

```bash
uv pip install 'crewai-tools[oci]'
```

## Usage

```python
from crewai_tools import OCIObjectStorageReaderTool, OCIObjectStorageWriterTool

reader = OCIObjectStorageReaderTool(namespace_name="my-namespace")
writer = OCIObjectStorageWriterTool(namespace_name="my-namespace")
```

Supported path formats:

- `oci://bucket/path/to/file.txt`
- `oci://namespace@bucket/path/to/file.txt`
