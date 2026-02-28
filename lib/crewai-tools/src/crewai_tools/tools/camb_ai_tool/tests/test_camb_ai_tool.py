"""Tests for CAMB AI crewAI tools."""

import json
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest

from crewai_tools.tools.camb_ai_tool.camb_ai_tool import (
    CambAIToolkit,
    CambAudioSeparationTool,
    CambTextToSoundTool,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambTTSTool,
    CambVoiceCloneTool,
    CambVoiceListTool,
    _detect_audio_format,
    _add_wav_header,
)


# ---------------------------------------------------------------------------
# Audio format detection
# ---------------------------------------------------------------------------


def test_detect_wav():
    assert _detect_audio_format(b"RIFF" + b"\x00" * 100) == "wav"


def test_detect_mp3():
    assert _detect_audio_format(b"\xff\xfb" + b"\x00" * 100) == "mp3"


def test_detect_flac():
    assert _detect_audio_format(b"fLaC" + b"\x00" * 100) == "flac"


def test_detect_ogg():
    assert _detect_audio_format(b"OggS" + b"\x00" * 100) == "ogg"


def test_detect_from_content_type():
    assert _detect_audio_format(b"\x00" * 100, "audio/mpeg") == "mp3"


def test_detect_unknown():
    assert _detect_audio_format(b"\x00" * 100) == "pcm"


def test_wav_header():
    pcm = b"\x00" * 100
    wav = _add_wav_header(pcm)
    assert wav.startswith(b"RIFF")
    assert b"WAVE" in wav[:12]
    assert wav.endswith(pcm)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
def test_tts(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.text_to_speech.tts.return_value = [b"chunk1", b"chunk2"]

    tool = CambTTSTool(api_key="test-key")
    result = tool._run(text="Hello world")

    assert result.endswith(".wav")
    mock_client.text_to_speech.tts.assert_called_once()


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
def test_tts_base64(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.text_to_speech.tts.return_value = [b"audio"]

    tool = CambTTSTool(api_key="test-key")
    result = tool._run(text="Hello", output_format="base64")

    import base64
    decoded = base64.b64decode(result)
    assert decoded == b"audio"


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
def test_translation(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_result = Mock()
    mock_result.text = "Hola mundo"
    mock_client.translation.translation_stream.return_value = mock_result

    tool = CambTranslationTool(api_key="test-key")
    result = tool._run(text="Hello world", source_language=1, target_language=2)

    assert result == "Hola mundo"


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
def test_translation_api_error_workaround(mock_get_client):
    from camb.core.api_error import ApiError

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.translation.translation_stream.side_effect = ApiError(
        status_code=200, body="Hola mundo"
    )

    tool = CambTranslationTool(api_key="test-key")
    result = tool._run(text="Hello world", source_language=1, target_language=2)

    assert result == "Hola mundo"


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


@patch("httpx.get")
@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
def test_transcription(mock_get_client, mock_httpx_get):
    mock_resp = Mock()
    mock_resp.content = b"fake audio data"
    mock_resp.raise_for_status = Mock()
    mock_httpx_get.return_value = mock_resp

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_create = Mock(task_id="task-1")
    mock_client.transcription.create_transcription.return_value = mock_create

    mock_status = Mock(status="completed", run_id="run-1")
    mock_client.transcription.get_transcription_task_status.return_value = mock_status

    mock_transcription = Mock(text="Hello world", segments=[], speakers=[])
    mock_client.transcription.get_transcription_result.return_value = mock_transcription

    tool = CambTranscriptionTool(api_key="test-key")
    result = tool._run(language=1, audio_url="https://example.com/audio.mp3")

    out = json.loads(result)
    assert out["text"] == "Hello world"


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
def test_transcription_no_source(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    tool = CambTranscriptionTool(api_key="test-key")
    result = tool._run(language=1)

    out = json.loads(result)
    assert "error" in out


# ---------------------------------------------------------------------------
# Voice List
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
def test_voice_list(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_voice = Mock(id=123, voice_name="Test Voice", gender=1, age=30, language=1)
    mock_client.voice_cloning.list_voices.return_value = [mock_voice]

    tool = CambVoiceListTool(api_key="test-key")
    result = tool._run()

    voices = json.loads(result)
    assert len(voices) == 1
    assert voices[0]["id"] == 123
    assert voices[0]["gender"] == "male"


# ---------------------------------------------------------------------------
# Voice Clone
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
def test_voice_clone(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_result = Mock(voice_id=999, message="Voice created")
    mock_client.voice_cloning.create_custom_voice.return_value = mock_result

    tool = CambVoiceCloneTool(api_key="test-key")

    with patch("builtins.open", mock_open(read_data=b"audio_data")):
        result = tool._run(voice_name="My Voice", audio_file_path="/fake/path.wav", gender=1)

    out = json.loads(result)
    assert out["voice_id"] == 999
    assert out["status"] == "created"


# ---------------------------------------------------------------------------
# Text to Sound
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._poll_task")
def test_text_to_sound(mock_poll, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_create = Mock(task_id="task-1")
    mock_client.text_to_audio.create_text_to_audio.return_value = mock_create

    mock_status = Mock(status="completed", run_id="run-1")
    mock_poll.return_value = mock_status

    mock_client.text_to_audio.get_text_to_audio_result.return_value = [b"audio"]

    tool = CambTextToSoundTool(api_key="test-key")
    result = tool._run(prompt="upbeat music")

    assert result.endswith(".wav")


# ---------------------------------------------------------------------------
# Audio Separation
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._get_camb_client")
@patch("crewai_tools.tools.camb_ai_tool.camb_ai_tool._poll_task")
def test_audio_separation(mock_poll, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_create = Mock(task_id="task-1")
    mock_client.audio_separation.create_audio_separation.return_value = mock_create

    mock_status = Mock(status="completed", run_id="run-1")
    mock_poll.return_value = mock_status

    mock_sep = Mock(
        vocals_url="https://example.com/vocals.wav",
        background_url="https://example.com/bg.wav",
        voice_url=None, instrumental_url=None, vocals=None, background=None,
    )
    mock_client.audio_separation.get_audio_separation_run_info.return_value = mock_sep

    tool = CambAudioSeparationTool(api_key="test-key")

    with patch("builtins.open", mock_open(read_data=b"audio")):
        result = tool._run(audio_file_path="/fake/audio.mp3")

    out = json.loads(result)
    assert out["status"] == "completed"
    assert out["vocals"] == "https://example.com/vocals.wav"


# ---------------------------------------------------------------------------
# Toolkit
# ---------------------------------------------------------------------------


def test_toolkit_get_tools():
    with patch.dict("os.environ", {"CAMB_API_KEY": "test-key"}):
        toolkit = CambAIToolkit()
        tools = toolkit.get_tools()
        assert len(tools) == 9


def test_toolkit_selective():
    with patch.dict("os.environ", {"CAMB_API_KEY": "test-key"}):
        toolkit = CambAIToolkit(include_tts=True, include_translation=True,
                                 include_transcription=False, include_translated_tts=False,
                                 include_voice_clone=False, include_voice_list=False,
                                 include_text_to_sound=False, include_audio_separation=False)
        tools = toolkit.get_tools()
        assert len(tools) == 3


def test_toolkit_no_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        toolkit = CambAIToolkit()
        with pytest.raises(ValueError, match="CAMB AI API key is required"):
            toolkit.get_tools()
