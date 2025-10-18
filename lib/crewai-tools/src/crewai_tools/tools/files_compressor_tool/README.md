# 📦 FileCompressorTool

The **FileCompressorTool** is a utility for compressing individual files or entire directories (including nested subdirectories) into different archive formats, such as `.zip` or `.tar` (including `.tar.gz`, `.tar.bz2`, and `.tar.xz`). This tool is useful for archiving logs, documents, datasets, or backups in a compact format, and ensures flexibility in how the archives are created.

---

## Description

This tool:
- Accepts a **file or directory** as input.
- Supports **recursive compression** of subdirectories.
- Lets you define a **custom output archive path** or defaults to the current directory.
- Handles **overwrite protection** to avoid unintentional data loss.
- Supports multiple compression formats: `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, and `.tar.xz`.

---

## Arguments

| Argument      | Type      | Required | Description                                                                 |
|---------------|-----------|----------|-----------------------------------------------------------------------------|
| `input_path`  | `str`     | ✅       | Path to the file or directory you want to compress.                         |
| `output_path` | `str`     | ❌       | Optional path for the resulting archive file. Defaults to `./<name>.<format>`. |
| `overwrite`   | `bool`    | ❌       | Whether to overwrite an existing archive file. Defaults to `False`.         |
| `format`      | `str`     | ❌       | Compression format to use. Can be one of `zip`, `tar`, `tar.gz`, `tar.bz2`, `tar.xz`. Defaults to `zip`. |

---


## Usage Example

```python
from crewai_tools import FileCompressorTool

# Initialize the tool
tool = FileCompressorTool()

# Compress a directory with subdirectories and files into a zip archive
result = tool._run(
    input_path="./data/project_docs",           # Folder containing subfolders & files
    output_path="./output/project_docs.zip",    # Optional output path (defaults to zip format)
    overwrite=True                              # Allow overwriting if file exists
)
print(result)
# Example output: Successfully compressed './data/project_docs' into './output/project_docs.zip'

```

---

## Example Scenarios

### Compress a single file into a zip archive:
```python
# Compress a single file into a zip archive
result = tool._run(input_path="report.pdf")
# Example output: Successfully compressed 'report.pdf' into './report.zip'
```

### Compress a directory with nested folders into a zip archive:
```python
# Compress a directory containing nested subdirectories and files
result = tool._run(input_path="./my_data", overwrite=True)
# Example output: Successfully compressed 'my_data' into './my_data.zip'
```

### Use a custom output path with a zip archive:
```python
# Compress a directory and specify a custom zip output location
result = tool._run(input_path="./my_data", output_path="./backups/my_data_backup.zip", overwrite=True)
# Example output: Successfully compressed 'my_data' into './backups/my_data_backup.zip'
```

### Prevent overwriting an existing zip file:
```python
# Try to compress a directory without overwriting an existing zip file
result = tool._run(input_path="./my_data", output_path="./backups/my_data_backup.zip", overwrite=False)
# Example output: Output zip './backups/my_data_backup.zip' already exists and overwrite is set to False.
```

### Compress into a tar archive:
```python
# Compress a directory into a tar archive
result = tool._run(input_path="./my_data", format="tar", overwrite=True)
# Example output: Successfully compressed 'my_data' into './my_data.tar'
```

### Compress into a tar.gz archive:
```python
# Compress a directory into a tar.gz archive
result = tool._run(input_path="./my_data", format="tar.gz", overwrite=True)
# Example output: Successfully compressed 'my_data' into './my_data.tar.gz'
```

### Compress into a tar.bz2 archive:
```python
# Compress a directory into a tar.bz2 archive
result = tool._run(input_path="./my_data", format="tar.bz2", overwrite=True)
# Example output: Successfully compressed 'my_data' into './my_data.tar.bz2'
```

### Compress into a tar.xz archive:
```python
# Compress a directory into a tar.xz archive
result = tool._run(input_path="./my_data", format="tar.xz", overwrite=True)
# Example output: Successfully compressed 'my_data' into './my_data.tar.xz'
```

---

## Error Handling and Validations

- **File Extension Validation**: The tool ensures that the output file extension matches the selected format (e.g., `.zip` for `zip` format, `.tar` for `tar` format, etc.).
- **File/Directory Existence**: If the input path does not exist, an error message will be returned.
- **Overwrite Protection**: If a file already exists at the output path, the tool checks the `overwrite` flag before proceeding. If `overwrite=False`, it prevents overwriting the existing file.

---

This tool provides a flexible and robust way to handle file and directory compression across multiple formats for efficient storage and backups.
