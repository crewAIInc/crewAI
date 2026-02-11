# crewai-files

File handling utilities for CrewAI multimodal inputs.

## Supported File Types

- `ImageFile` - PNG, JPEG, GIF, WebP
- `PDFFile` - PDF documents
- `TextFile` - Plain text files
- `AudioFile` - MP3, WAV, FLAC, OGG, M4A
- `VideoFile` - MP4, WebM, MOV, AVI

## Usage

```python
from crewai_files import File, ImageFile, PDFFile

# Auto-detect file type
file = File(source="document.pdf")  # Resolves to PDFFile

# Or use specific types
image = ImageFile(source="chart.png")
pdf = PDFFile(source="report.pdf")
```

### Passing Files to Crews

```python
crew.kickoff(
    input_files={"chart": ImageFile(source="chart.png")}
)
```

### Passing Files to Tasks

```python
task = Task(
    description="Analyze the chart",
    expected_output="Analysis",
    agent=agent,
    input_files=[ImageFile(source="chart.png")],
)
```
