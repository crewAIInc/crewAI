"""Integration tests for Agent multimodal functionality with input_files.

Tests agent.kickoff(input_files={...}) across different providers and file types.
"""

from pathlib import Path

import pytest

from crewai import Agent, LLM
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

OPENAI_IMAGE_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/o4-mini",
]

OPENAI_RESPONSES_MODELS = [
    ("openai/gpt-4o-mini", "responses"),
    ("openai/o4-mini", "responses"),
]

ANTHROPIC_MODELS = [
    "anthropic/claude-3-5-haiku-20241022",
]

GEMINI_MODELS = [
    "gemini/gemini-2.0-flash",
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


def _create_analyst_agent(llm: LLM) -> Agent:
    """Create a simple analyst agent for file analysis."""
    return Agent(
        role="File Analyst",
        goal="Analyze and describe files accurately",
        backstory="Expert at analyzing various file types.",
        llm=llm,
        verbose=False,
    )


class TestAgentMultimodalOpenAI:
    """Test Agent with input_files using OpenAI models."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", OPENAI_IMAGE_MODELS)
    def test_image_file(self, model: str, image_file: ImageFile) -> None:
        """Test agent can process an image file."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this image briefly."}],
            input_files={"chart": image_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", OPENAI_IMAGE_MODELS)
    def test_image_bytes(self, model: str, image_bytes: bytes) -> None:
        """Test agent can process image bytes."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this image briefly."}],
            input_files={"chart": ImageFile(source=image_bytes)},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", OPENAI_IMAGE_MODELS)
    def test_generic_file_image(self, model: str, image_bytes: bytes) -> None:
        """Test agent can process generic File with auto-detected image."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this image briefly."}],
            input_files={"chart": File(source=image_bytes)},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0


class TestAgentMultimodalOpenAIResponses:
    """Test Agent with input_files using OpenAI Responses API."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model,api", OPENAI_RESPONSES_MODELS)
    def test_image_file(
        self, model: str, api: str, image_file: ImageFile
    ) -> None:
        """Test agent can process an image file with Responses API."""
        llm = LLM(model=model, api=api)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this image briefly."}],
            input_files={"chart": image_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model,api", OPENAI_RESPONSES_MODELS)
    def test_pdf_file(self, model: str, api: str, pdf_file: PDFFile) -> None:
        """Test agent can process a PDF file with Responses API."""
        llm = LLM(model=model, api=api)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "What type of document is this?"}],
            input_files={"document": pdf_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0


class TestAgentMultimodalAnthropic:
    """Test Agent with input_files using Anthropic models."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", ANTHROPIC_MODELS)
    def test_image_file(self, model: str, image_file: ImageFile) -> None:
        """Test agent can process an image file."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this image briefly."}],
            input_files={"chart": image_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", ANTHROPIC_MODELS)
    def test_pdf_file(self, model: str, pdf_file: PDFFile) -> None:
        """Test agent can process a PDF file."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "What type of document is this?"}],
            input_files={"document": pdf_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", ANTHROPIC_MODELS)
    def test_mixed_files(
        self, model: str, image_file: ImageFile, pdf_file: PDFFile
    ) -> None:
        """Test agent can process multiple file types together."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "What files do you see?"}],
            input_files={"chart": image_file, "document": pdf_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0


class TestAgentMultimodalGemini:
    """Test Agent with input_files using Gemini models."""

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_image_file(self, model: str, image_file: ImageFile) -> None:
        """Test agent can process an image file."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this image briefly."}],
            input_files={"chart": image_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_text_file(self, model: str, text_file: TextFile) -> None:
        """Test agent can process a text file."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "Summarize this text briefly."}],
            input_files={"readme": text_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_video_file(self, model: str, video_file: VideoFile) -> None:
        """Test agent can process a video file."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "What do you see in this video?"}],
            input_files={"video": video_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_audio_file(self, model: str, audio_file: AudioFile) -> None:
        """Test agent can process an audio file."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "What do you hear in this audio?"}],
            input_files={"audio": audio_file},
        )

        assert result
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
        """Test agent can process multiple file types together."""
        llm = LLM(model=model)
        agent = _create_analyst_agent(llm)

        result = agent.kickoff(
            messages=[{"role": "user", "content": "What files do you see?"}],
            input_files={"chart": image_file, "readme": text_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0


class TestAgentMultimodalFileTypes:
    """Test all file types with appropriate providers."""

    @pytest.mark.vcr()
    def test_image_openai(self, image_file: ImageFile) -> None:
        """Test image file with OpenAI."""
        llm = LLM(model="openai/gpt-4o-mini")
        agent = _create_analyst_agent(llm)
        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this image."}],
            input_files={"image": image_file},
        )
        assert result.raw

    @pytest.mark.vcr()
    def test_pdf_anthropic(self, pdf_file: PDFFile) -> None:
        """Test PDF file with Anthropic."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        agent = _create_analyst_agent(llm)
        result = agent.kickoff(
            messages=[{"role": "user", "content": "What is this document?"}],
            input_files={"document": pdf_file},
        )
        assert result.raw

    @pytest.mark.vcr()
    def test_pdf_openai_responses(self, pdf_file: PDFFile) -> None:
        """Test PDF file with OpenAI Responses API."""
        llm = LLM(model="openai/gpt-4o-mini", api="responses")
        agent = _create_analyst_agent(llm)
        result = agent.kickoff(
            messages=[{"role": "user", "content": "What is this document?"}],
            input_files={"document": pdf_file},
        )
        assert result.raw

    @pytest.mark.vcr()
    def test_text_gemini(self, text_file: TextFile) -> None:
        """Test text file with Gemini."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        agent = _create_analyst_agent(llm)
        result = agent.kickoff(
            messages=[{"role": "user", "content": "Summarize this text."}],
            input_files={"readme": text_file},
        )
        assert result.raw

    @pytest.mark.vcr()
    def test_video_gemini(self, video_file: VideoFile) -> None:
        """Test video file with Gemini."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        agent = _create_analyst_agent(llm)
        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this video."}],
            input_files={"video": video_file},
        )
        assert result.raw

    @pytest.mark.vcr()
    def test_audio_gemini(self, audio_file: AudioFile) -> None:
        """Test audio file with Gemini."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        agent = _create_analyst_agent(llm)
        result = agent.kickoff(
            messages=[{"role": "user", "content": "Describe this audio."}],
            input_files={"audio": audio_file},
        )
        assert result.raw


class TestAgentMultimodalAsync:
    """Test async agent execution with files."""

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_async_agent_with_image(self, image_file: ImageFile) -> None:
        """Test async agent with image file."""
        llm = LLM(model="openai/gpt-4o-mini")
        agent = _create_analyst_agent(llm)

        result = await agent.kickoff_async(
            messages=[{"role": "user", "content": "Describe this image."}],
            input_files={"chart": image_file},
        )

        assert result
        assert result.raw
        assert len(result.raw) > 0