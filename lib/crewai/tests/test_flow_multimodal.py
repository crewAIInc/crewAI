"""Integration tests for Flow multimodal functionality with input_files.

Tests flow.kickoff(input_files={...}) with crews that process files.
"""

from pathlib import Path

import pytest

from crewai import Agent, Crew, LLM, Task
from crewai.flow.flow import Flow, listen, start
from crewai_files import AudioFile, File, ImageFile, PDFFile, TextFile, VideoFile


TEST_FIXTURES_DIR = (
    Path(__file__).parent.parent.parent / "crewai-files" / "tests" / "fixtures"
)
TEST_IMAGE_PATH = TEST_FIXTURES_DIR / "revenue_chart.png"
TEST_TEXT_PATH = TEST_FIXTURES_DIR / "review_guidelines.txt"
TEST_VIDEO_PATH = TEST_FIXTURES_DIR / "sample_video.mp4"
TEST_AUDIO_PATH = TEST_FIXTURES_DIR / "sample_audio.wav"

MINIMAL_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer << /Size 4 /Root 1 0 R >>
startxref
196
%%EOF
"""


@pytest.fixture
def image_file() -> ImageFile:
    """Create an ImageFile from test fixture."""
    return ImageFile(source=str(TEST_IMAGE_PATH))


@pytest.fixture
def image_bytes() -> bytes:
    """Load test image bytes."""
    return TEST_IMAGE_PATH.read_bytes()


@pytest.fixture
def text_file() -> TextFile:
    """Create a TextFile from test fixture."""
    return TextFile(source=str(TEST_TEXT_PATH))


@pytest.fixture
def pdf_file() -> PDFFile:
    """Create a PDFFile from minimal PDF bytes."""
    return PDFFile(source=MINIMAL_PDF)


@pytest.fixture
def video_file() -> VideoFile:
    """Create a VideoFile from test fixture."""
    if not TEST_VIDEO_PATH.exists():
        pytest.skip("sample_video.mp4 fixture not found")
    return VideoFile(source=str(TEST_VIDEO_PATH))


@pytest.fixture
def audio_file() -> AudioFile:
    """Create an AudioFile from test fixture."""
    if not TEST_AUDIO_PATH.exists():
        pytest.skip("sample_audio.wav fixture not found")
    return AudioFile(source=str(TEST_AUDIO_PATH))


def _create_analyst_crew(llm: LLM) -> Crew:
    """Create a simple analyst crew for file analysis."""
    agent = Agent(
        role="File Analyst",
        goal="Analyze and describe files accurately",
        backstory="Expert at analyzing various file types.",
        llm=llm,
        verbose=False,
    )
    task = Task(
        description="Describe the file(s) you see. Be brief, one sentence max.",
        expected_output="A brief description of the file.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], verbose=False)


class TestFlowMultimodalOpenAI:
    """Test Flow with input_files using OpenAI models."""

    @pytest.mark.vcr()
    def test_flow_with_image_file(self, image_file: ImageFile) -> None:
        """Test flow passes input_files to crew."""

        class ImageAnalysisFlow(Flow):
            @start()
            def analyze_image(self) -> str:
                llm = LLM(model="openai/gpt-4o-mini")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = ImageAnalysisFlow()
        result = flow.kickoff(input_files={"chart": image_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.vcr()
    def test_flow_with_image_bytes(self, image_bytes: bytes) -> None:
        """Test flow with image bytes."""

        class ImageAnalysisFlow(Flow):
            @start()
            def analyze_image(self) -> str:
                llm = LLM(model="openai/gpt-4o-mini")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = ImageAnalysisFlow()
        result = flow.kickoff(input_files={"chart": ImageFile(source=image_bytes)})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0


class TestFlowMultimodalAnthropic:
    """Test Flow with input_files using Anthropic models."""

    @pytest.mark.vcr()
    def test_flow_with_image_file(self, image_file: ImageFile) -> None:
        """Test flow passes input_files to crew."""

        class ImageAnalysisFlow(Flow):
            @start()
            def analyze_image(self) -> str:
                llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = ImageAnalysisFlow()
        result = flow.kickoff(input_files={"chart": image_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.vcr()
    def test_flow_with_pdf_file(self, pdf_file: PDFFile) -> None:
        """Test flow with PDF file."""

        class PDFAnalysisFlow(Flow):
            @start()
            def analyze_pdf(self) -> str:
                llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = PDFAnalysisFlow()
        result = flow.kickoff(input_files={"document": pdf_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0


class TestFlowMultimodalGemini:
    """Test Flow with input_files using Gemini models."""

    @pytest.mark.vcr()
    def test_flow_with_image_file(self, image_file: ImageFile) -> None:
        """Test flow with image file."""

        class ImageAnalysisFlow(Flow):
            @start()
            def analyze_image(self) -> str:
                llm = LLM(model="gemini/gemini-2.0-flash")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = ImageAnalysisFlow()
        result = flow.kickoff(input_files={"chart": image_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.vcr()
    def test_flow_with_text_file(self, text_file: TextFile) -> None:
        """Test flow with text file."""

        class TextAnalysisFlow(Flow):
            @start()
            def analyze_text(self) -> str:
                llm = LLM(model="gemini/gemini-2.0-flash")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = TextAnalysisFlow()
        result = flow.kickoff(input_files={"readme": text_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.vcr()
    def test_flow_with_video_file(self, video_file: VideoFile) -> None:
        """Test flow with video file."""

        class VideoAnalysisFlow(Flow):
            @start()
            def analyze_video(self) -> str:
                llm = LLM(model="gemini/gemini-2.0-flash")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = VideoAnalysisFlow()
        result = flow.kickoff(input_files={"video": video_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.vcr()
    def test_flow_with_audio_file(self, audio_file: AudioFile) -> None:
        """Test flow with audio file."""

        class AudioAnalysisFlow(Flow):
            @start()
            def analyze_audio(self) -> str:
                llm = LLM(model="gemini/gemini-2.0-flash")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = AudioAnalysisFlow()
        result = flow.kickoff(input_files={"audio": audio_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0


class TestFlowMultimodalMultiStep:
    """Test multi-step flows with file processing."""

    @pytest.mark.vcr()
    def test_flow_with_multiple_crews(self, image_file: ImageFile) -> None:
        """Test flow passes files through multiple crews."""

        class MultiStepFlow(Flow):
            @start()
            def describe_image(self) -> str:
                llm = LLM(model="openai/gpt-4o-mini")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

            @listen(describe_image)
            def summarize_description(self, description: str) -> str:
                llm = LLM(model="openai/gpt-4o-mini")
                agent = Agent(
                    role="Summarizer",
                    goal="Summarize text concisely",
                    backstory="Expert at summarization.",
                    llm=llm,
                    verbose=False,
                )
                task = Task(
                    description=f"Summarize this in 5 words: {description}",
                    expected_output="A 5-word summary.",
                    agent=agent,
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                result = crew.kickoff()
                return result.raw

        flow = MultiStepFlow()
        result = flow.kickoff(input_files={"chart": image_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.vcr()
    def test_flow_with_mixed_files(
        self, image_file: ImageFile, text_file: TextFile
    ) -> None:
        """Test flow with multiple file types."""

        class MixedFilesFlow(Flow):
            @start()
            def analyze_files(self) -> str:
                llm = LLM(model="gemini/gemini-2.0-flash")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = MixedFilesFlow()
        result = flow.kickoff(
            input_files={"chart": image_file, "readme": text_file}
        )

        assert result
        assert isinstance(result, str)
        assert len(result) > 0


class TestFlowMultimodalAsync:
    """Test async flow execution with files."""

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_async_flow_with_image(self, image_file: ImageFile) -> None:
        """Test async flow with image file."""

        class AsyncImageFlow(Flow):
            @start()
            def analyze_image(self) -> str:
                llm = LLM(model="openai/gpt-4o-mini")
                crew = _create_analyst_crew(llm)
                result = crew.kickoff()
                return result.raw

        flow = AsyncImageFlow()
        result = await flow.kickoff_async(input_files={"chart": image_file})

        assert result
        assert isinstance(result, str)
        assert len(result) > 0