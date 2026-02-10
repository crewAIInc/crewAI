"""Integration tests for Crew multimodal functionality with input_files.

Tests crew.kickoff(input_files={...}) across different providers and file types.
"""

from pathlib import Path

import pytest

from crewai import Agent, Crew, LLM, Task
from crewai_files import AudioFile, File, ImageFile, PDFFile, TextFile, VideoFile


TEST_FIXTURES_DIR = (
    Path(__file__).parent.parent.parent / "crewai-files" / "tests" / "fixtures"
)
TEST_IMAGE_PATH = TEST_FIXTURES_DIR / "revenue_chart.png"
TEST_TEXT_PATH = TEST_FIXTURES_DIR / "review_guidelines.txt"
TEST_VIDEO_PATH = TEST_FIXTURES_DIR / "sample_video.mp4"
TEST_AUDIO_PATH = TEST_FIXTURES_DIR / "sample_audio.wav"
TEST_PDF_PATH = TEST_FIXTURES_DIR / "agents.pdf"

OPENAI_IMAGE_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/o4-mini",
    "openai/gpt-4.1-mini",
]

OPENAI_RESPONSES_MODELS = [
    ("openai/gpt-4o-mini", "responses"),
    ("openai/o4-mini", "responses"),
]

ANTHROPIC_MODELS = [
    "anthropic/claude-3-5-haiku-20241022",
    "anthropic/claude-sonnet-4-20250514",
]

GEMINI_MODELS = [
    "gemini/gemini-2.0-flash",
]

BEDROCK_MODELS = [
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
]


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
def text_bytes() -> bytes:
    """Load test text bytes."""
    return TEST_TEXT_PATH.read_bytes()


@pytest.fixture
def pdf_file() -> PDFFile:
    """Create a PDFFile from test fixture."""
    return PDFFile(source=str(TEST_PDF_PATH))


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


class TestCrewMultimodalOpenAI:
    """Test Crew with input_files using OpenAI models."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", OPENAI_IMAGE_MODELS)
    def test_image_file(self, model: str, image_file: ImageFile) -> None:
        """Test crew can process an image file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": image_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", OPENAI_IMAGE_MODELS)
    def test_image_bytes(self, model: str, image_bytes: bytes) -> None:
        """Test crew can process image bytes."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": ImageFile(source=image_bytes)})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", OPENAI_IMAGE_MODELS)
    def test_generic_file_image(self, model: str, image_bytes: bytes) -> None:
        """Test crew can process generic File with auto-detected image."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": File(source=image_bytes)})

        assert result.raw
        assert len(result.raw) > 0


class TestCrewMultimodalOpenAIResponses:
    """Test Crew with input_files using OpenAI Responses API."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model,api", OPENAI_RESPONSES_MODELS)
    def test_image_file(
        self, model: str, api: str, image_file: ImageFile
    ) -> None:
        """Test crew can process an image file with Responses API."""
        llm = LLM(model=model, api=api)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": image_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model,api", OPENAI_RESPONSES_MODELS)
    def test_pdf_file(self, model: str, api: str, pdf_file: PDFFile) -> None:
        """Test crew can process a PDF file with Responses API."""
        llm = LLM(model=model, api=api)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"document": pdf_file})

        assert result.raw
        assert len(result.raw) > 0


class TestCrewMultimodalAnthropic:
    """Test Crew with input_files using Anthropic models."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", ANTHROPIC_MODELS)
    def test_image_file(self, model: str, image_file: ImageFile) -> None:
        """Test crew can process an image file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": image_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", ANTHROPIC_MODELS)
    def test_pdf_file(self, model: str, pdf_file: PDFFile) -> None:
        """Test crew can process a PDF file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"document": pdf_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", ANTHROPIC_MODELS)
    def test_mixed_files(
        self, model: str, image_file: ImageFile, pdf_file: PDFFile
    ) -> None:
        """Test crew can process multiple file types together."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(
            input_files={"chart": image_file, "document": pdf_file}
        )

        assert result.raw
        assert len(result.raw) > 0


class TestCrewMultimodalGemini:
    """Test Crew with input_files using Gemini models."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_image_file(self, model: str, image_file: ImageFile) -> None:
        """Test crew can process an image file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": image_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_text_file(self, model: str, text_file: TextFile) -> None:
        """Test crew can process a text file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"readme": text_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_video_file(self, model: str, video_file: VideoFile) -> None:
        """Test crew can process a video file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"video": video_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_audio_file(self, model: str, audio_file: AudioFile) -> None:
        """Test crew can process an audio file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"audio": audio_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_mixed_files(
        self,
        model: str,
        image_file: ImageFile,
        text_file: TextFile,
    ) -> None:
        """Test crew can process multiple file types together."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(
            input_files={"chart": image_file, "readme": text_file}
        )

        assert result.raw
        assert len(result.raw) > 0


class TestCrewMultimodalBedrock:
    """Test Crew with input_files using Bedrock models."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", BEDROCK_MODELS)
    def test_image_file(self, model: str, image_file: ImageFile) -> None:
        """Test crew can process an image file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": image_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", BEDROCK_MODELS)
    def test_pdf_file(self, model: str, pdf_file: PDFFile) -> None:
        """Test crew can process a PDF file."""
        llm = LLM(model=model)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"document": pdf_file})

        assert result.raw
        assert len(result.raw) > 0


class TestCrewMultimodalFileTypes:
    """Test all file types with appropriate providers."""

    @pytest.mark.vcr()
    def test_image_openai(self, image_file: ImageFile) -> None:
        """Test image file with OpenAI."""
        llm = LLM(model="openai/gpt-4o-mini")
        crew = _create_analyst_crew(llm)
        result = crew.kickoff(input_files={"image": image_file})
        assert result.raw

    @pytest.mark.vcr()
    def test_pdf_anthropic(self, pdf_file: PDFFile) -> None:
        """Test PDF file with Anthropic."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        crew = _create_analyst_crew(llm)
        result = crew.kickoff(input_files={"document": pdf_file})
        assert result.raw

    @pytest.mark.vcr()
    def test_pdf_openai_responses(self, pdf_file: PDFFile) -> None:
        """Test PDF file with OpenAI Responses API."""
        llm = LLM(model="openai/gpt-4o-mini", api="responses")
        crew = _create_analyst_crew(llm)
        result = crew.kickoff(input_files={"document": pdf_file})
        assert result.raw

    @pytest.mark.vcr()
    def test_text_gemini(self, text_file: TextFile) -> None:
        """Test text file with Gemini."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        crew = _create_analyst_crew(llm)
        result = crew.kickoff(input_files={"readme": text_file})
        assert result.raw

    @pytest.mark.vcr()
    def test_video_gemini(self, video_file: VideoFile) -> None:
        """Test video file with Gemini."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        crew = _create_analyst_crew(llm)
        result = crew.kickoff(input_files={"video": video_file})
        assert result.raw

    @pytest.mark.vcr()
    def test_audio_gemini(self, audio_file: AudioFile) -> None:
        """Test audio file with Gemini."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        crew = _create_analyst_crew(llm)
        result = crew.kickoff(input_files={"audio": audio_file})
        assert result.raw


class TestCrewMultimodalUnsupportedTypes:
    """Test that unsupported file types fall back to read_file tool."""

    @pytest.mark.vcr()
    def test_video_with_openai_uses_tool(self, video_file: VideoFile) -> None:
        """Test video with OpenAI (no video support) uses read_file tool."""
        llm = LLM(model="openai/gpt-4o-mini")
        agent = Agent(
            role="File Analyst",
            goal="Analyze files",
            backstory="Expert analyst.",
            llm=llm,
            verbose=False,
        )
        task = Task(
            description="What type of file is this? Just name the file type.",
            expected_output="The file type.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        result = crew.kickoff(input_files={"video": video_file})

        assert result.raw
        # Should mention video or the filename since it can't directly process it

    @pytest.mark.vcr()
    def test_audio_with_anthropic_uses_tool(self, audio_file: AudioFile) -> None:
        """Test audio with Anthropic (no audio support) uses read_file tool."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        agent = Agent(
            role="File Analyst",
            goal="Analyze files",
            backstory="Expert analyst.",
            llm=llm,
            verbose=False,
        )
        task = Task(
            description="What type of file is this? Just name the file type.",
            expected_output="The file type.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        result = crew.kickoff(input_files={"audio": audio_file})

        assert result.raw


class TestCrewMultimodalFileUpload:
    """Test file upload functionality with prefer_upload=True."""

    @pytest.mark.vcr()
    def test_image_upload_anthropic(self, image_file: ImageFile) -> None:
        """Test image upload to Anthropic Files API."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022", prefer_upload=True)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": image_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    def test_image_upload_openai_responses(self, image_file: ImageFile) -> None:
        """Test image upload to OpenAI Files API via Responses API."""
        llm = LLM(model="openai/gpt-4o-mini", api="responses", prefer_upload=True)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"chart": image_file})

        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    def test_pdf_upload_anthropic(self, pdf_file: PDFFile) -> None:
        """Test PDF upload to Anthropic Files API."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022", prefer_upload=True)
        crew = _create_analyst_crew(llm)

        result = crew.kickoff(input_files={"document": pdf_file})

        assert result.raw
        assert len(result.raw) > 0