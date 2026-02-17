"""CAMB AI tools for crewAI.

Provides 9 audio/speech tools powered by CAMB AI:
- Text-to-Speech (TTS)
- Translation
- Transcription
- Translated TTS
- Voice Cloning
- Voice Listing
- Voice Creation from Description
- Text-to-Sound generation
- Audio Separation
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import tempfile
import time
from typing import Any, ClassVar, Literal, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# SDK client helpers (shared across tools)
# ---------------------------------------------------------------------------

def _get_camb_client(api_key: str, base_url: Optional[str], timeout: float):
    """Create a synchronous CambAI client."""
    from camb.client import CambAI

    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return CambAI(**kwargs)


def _poll_task(client_method, task_id, *, max_attempts=60, interval=2.0):
    """Poll an async CAMB AI task until completion."""
    _SUCCESS = {"completed", "SUCCESS"}
    _FAILURE = {"failed", "FAILED", "error", "ERROR", "TIMEOUT", "PAYMENT_REQUIRED"}
    for _ in range(max_attempts):
        status = client_method(task_id)
        if hasattr(status, "status"):
            val = status.status
            if val in _SUCCESS:
                return status
            if val in _FAILURE:
                raise RuntimeError(f"Task failed: {getattr(status, 'error', val)}")
        time.sleep(interval)
    raise TimeoutError(f"Task {task_id} did not complete within {max_attempts * interval}s")


def _detect_audio_format(audio_data: bytes, content_type: str = "") -> str:
    """Detect audio format from magic bytes and content-type."""
    if audio_data.startswith(b"RIFF"):
        return "wav"
    if audio_data.startswith((b"\xff\xfb", b"\xff\xfa", b"ID3")):
        return "mp3"
    if audio_data.startswith(b"fLaC"):
        return "flac"
    if audio_data.startswith(b"OggS"):
        return "ogg"
    ct = content_type.lower()
    if "wav" in ct or "wave" in ct:
        return "wav"
    if "mpeg" in ct or "mp3" in ct:
        return "mp3"
    if "flac" in ct:
        return "flac"
    if "ogg" in ct:
        return "ogg"
    return "pcm"


def _add_wav_header(pcm_data: bytes) -> bytes:
    """Add WAV header to raw PCM data (16-bit, 24 kHz, mono)."""
    sr, ch, bps = 24000, 1, 16
    byte_rate = sr * ch * bps // 8
    block_align = ch * bps // 8
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1,
        ch, sr, byte_rate, block_align, bps, b"data", data_size,
    )
    return header + pcm_data


def _save_audio(audio_data: bytes, suffix: str = ".wav") -> str:
    """Write audio bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        return f.name


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------

class TTSInput(BaseModel):
    text: str = Field(..., min_length=3, max_length=3000, description="Text to convert to speech (3-3000 characters).")
    language: str = Field(default="en-us", description="BCP-47 language code (e.g., 'en-us', 'es-es').")
    voice_id: int = Field(default=147320, description="Voice ID. Get available voices with CambVoiceListTool.")
    speech_model: str = Field(default="mars-flash", description="Speech model: 'mars-flash', 'mars-pro', or 'mars-instruct'.")
    output_format: Literal["file_path", "base64"] = Field(default="file_path", description="Output format.")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier (0.5-2.0).")
    user_instructions: Optional[str] = Field(default=None, description="Instructions for mars-instruct model.")


class TranslationInput(BaseModel):
    text: str = Field(..., description="Text to translate.")
    source_language: int = Field(..., description="Source language code. 1=English, 2=Spanish, 3=French, 4=German, 5=Italian.")
    target_language: int = Field(..., description="Target language code.")
    formality: Optional[int] = Field(default=None, description="Formality: 1=formal, 2=informal.")


class TranscriptionInput(BaseModel):
    language: int = Field(..., description="Language code. 1=English, 2=Spanish, 3=French, etc.")
    audio_url: Optional[str] = Field(default=None, description="URL of audio file.")
    audio_file_path: Optional[str] = Field(default=None, description="Local path to audio file.")

    @model_validator(mode="after")
    def validate_audio_source(self) -> "TranscriptionInput":
        if not self.audio_url and not self.audio_file_path:
            raise ValueError("Either audio_url or audio_file_path must be provided.")
        if self.audio_url and self.audio_file_path:
            raise ValueError("Provide only one of audio_url or audio_file_path.")
        return self


class TranslatedTTSInput(BaseModel):
    text: str = Field(..., description="Text to translate and convert to speech.")
    source_language: int = Field(..., description="Source language code.")
    target_language: int = Field(..., description="Target language code.")
    voice_id: int = Field(default=147320, description="Voice ID for TTS output.")
    output_format: Literal["file_path", "base64"] = Field(default="file_path", description="Output format.")
    formality: Optional[int] = Field(default=None, description="Formality: 1=formal, 2=informal.")


class VoiceCloneInput(BaseModel):
    voice_name: str = Field(..., description="Name for the new cloned voice.")
    audio_file_path: str = Field(..., description="Path to audio file (2+ seconds).")
    gender: int = Field(..., description="Gender: 1=Male, 2=Female, 0=Not Specified, 9=Not Applicable.")
    description: Optional[str] = Field(default=None, description="Optional voice description.")
    age: Optional[int] = Field(default=None, description="Optional age of the voice.")
    language: Optional[int] = Field(default=None, description="Optional language code.")


class VoiceListInput(BaseModel):
    pass


class TextToSoundInput(BaseModel):
    prompt: str = Field(..., description="Description of the sound or music to generate.")
    duration: Optional[float] = Field(default=None, description="Duration in seconds.")
    audio_type: Optional[Literal["music", "sound"]] = Field(default=None, description="Type: 'music' or 'sound'.")
    output_format: Literal["file_path", "base64"] = Field(default="file_path", description="Output format.")


class VoiceFromDescriptionInput(BaseModel):
    text: str = Field(..., description="Sample text the generated voice will speak.")
    voice_description: str = Field(..., min_length=100, description="Detailed description of the desired voice (minimum 100 characters / 18+ words). Include accent, tone, age, gender, etc.")


class AudioSeparationInput(BaseModel):
    audio_url: Optional[str] = Field(default=None, description="URL of audio file.")
    audio_file_path: Optional[str] = Field(default=None, description="Local path to audio file.")

    @model_validator(mode="after")
    def validate_audio_source(self) -> "AudioSeparationInput":
        if not self.audio_url and not self.audio_file_path:
            raise ValueError("Either audio_url or audio_file_path must be provided.")
        if self.audio_url and self.audio_file_path:
            raise ValueError("Provide only one of audio_url or audio_file_path.")
        return self


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_COMMON_ENV_VARS = [
    EnvVar(name="CAMB_API_KEY", description="API key for CAMB AI services", required=True),
]


class CambTTSTool(BaseTool):
    """Convert text to speech using CAMB AI. Supports 140+ languages and multiple voice models."""

    name: str = "camb_tts"
    description: str = (
        "Convert text to speech using CAMB AI. "
        "Supports 140+ languages and multiple voice models. "
        "Returns audio as a file path or base64 string."
    )
    args_schema: type[BaseModel] = TTSInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        text: str,
        language: str = "en-us",
        voice_id: int = 147320,
        speech_model: str = "mars-flash",
        output_format: str = "file_path",
        speed: float = 1.0,
        user_instructions: Optional[str] = None,
    ) -> str:
        from camb import StreamTtsOutputConfiguration, StreamTtsVoiceSettings

        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)

        kwargs: dict[str, Any] = {
            "text": text, "language": language, "voice_id": voice_id,
            "speech_model": speech_model,
            "output_configuration": StreamTtsOutputConfiguration(format="wav"),
            "voice_settings": StreamTtsVoiceSettings(speed=speed),
        }
        if user_instructions and speech_model == "mars-instruct":
            kwargs["user_instructions"] = user_instructions

        audio_chunks: list[bytes] = []
        for chunk in client.text_to_speech.tts(**kwargs):
            audio_chunks.append(chunk)
        audio_data = b"".join(audio_chunks)

        if output_format == "base64":
            return base64.b64encode(audio_data).decode("utf-8")
        return _save_audio(audio_data, ".wav")


class CambTranslationTool(BaseTool):
    """Translate text between 140+ languages using CAMB AI."""

    name: str = "camb_translation"
    description: str = (
        "Translate text between 140+ languages using CAMB AI. "
        "Common language codes: 1=English, 2=Spanish, 3=French, 4=German, 5=Italian."
    )
    args_schema: type[BaseModel] = TranslationInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        text: str,
        source_language: int,
        target_language: int,
        formality: Optional[int] = None,
    ) -> str:
        from camb.core.api_error import ApiError

        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)

        kwargs: dict[str, Any] = {
            "text": text, "source_language": source_language,
            "target_language": target_language,
        }
        if formality:
            kwargs["formality"] = formality

        try:
            result = client.translation.translation_stream(**kwargs)
            return self._extract_text(result)
        except ApiError as e:
            if e.status_code == 200 and e.body:
                return str(e.body)
            raise

    @staticmethod
    def _extract_text(result) -> str:
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            chunks = []
            for chunk in result:
                if hasattr(chunk, "text"):
                    chunks.append(chunk.text)
                elif isinstance(chunk, str):
                    chunks.append(chunk)
            return "".join(chunks)
        if hasattr(result, "text"):
            return result.text
        if isinstance(result, str):
            return result
        return str(result)


class CambTranscriptionTool(BaseTool):
    """Transcribe audio to text with speaker identification using CAMB AI."""

    name: str = "camb_transcription"
    description: str = (
        "Transcribe audio to text using CAMB AI. "
        "Supports audio URLs or local files. "
        "Returns transcription with segments and speaker identification."
    )
    args_schema: type[BaseModel] = TranscriptionInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        language: int,
        audio_url: Optional[str] = None,
        audio_file_path: Optional[str] = None,
    ) -> str:
        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)

        kwargs: dict[str, Any] = {"language": language}

        if audio_url:
            import httpx

            resp = httpx.get(audio_url)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            try:
                with open(tmp_path, "rb") as f:
                    kwargs["media_file"] = f
                    result = client.transcription.create_transcription(**kwargs)
            finally:
                os.unlink(tmp_path)
        elif audio_file_path:
            with open(audio_file_path, "rb") as f:
                kwargs["media_file"] = f
                result = client.transcription.create_transcription(**kwargs)
        else:
            return json.dumps({"error": "No audio source provided"})

        task_id = result.task_id
        status = _poll_task(client.transcription.get_transcription_task_status, task_id)
        run_id = status.run_id
        transcription = client.transcription.get_transcription_result(run_id)

        out: dict[str, Any] = {"text": getattr(transcription, "text", ""), "segments": [], "speakers": []}
        if hasattr(transcription, "segments"):
            for seg in transcription.segments:
                out["segments"].append({
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "text": getattr(seg, "text", ""),
                    "speaker": getattr(seg, "speaker", None),
                })
        if hasattr(transcription, "speakers"):
            out["speakers"] = list(transcription.speakers)
        elif out["segments"]:
            out["speakers"] = list({s["speaker"] for s in out["segments"] if s.get("speaker")})
        return json.dumps(out, indent=2)


class CambTranslatedTTSTool(BaseTool):
    """Translate text and convert to speech in one step using CAMB AI."""

    name: str = "camb_translated_tts"
    description: str = (
        "Translate text and convert to speech in one step. "
        "Provide source text, source language, target language, and voice ID. "
        "Returns audio file of the translated text spoken in the target language."
    )
    args_schema: type[BaseModel] = TranslatedTTSInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        text: str,
        source_language: int,
        target_language: int,
        voice_id: int = 147320,
        output_format: str = "file_path",
        formality: Optional[int] = None,
    ) -> str:
        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)

        kwargs: dict[str, Any] = {
            "text": text, "voice_id": voice_id,
            "source_language": source_language, "target_language": target_language,
        }
        if formality:
            kwargs["formality"] = formality

        result = client.translated_tts.create_translated_tts(**kwargs)
        task_id = result.task_id
        status = _poll_task(client.translated_tts.get_translated_tts_task_status, task_id)

        audio_data, audio_fmt = self._get_audio(status, key, client)

        if audio_fmt == "pcm" and audio_data:
            audio_data = _add_wav_header(audio_data)
            audio_fmt = "wav"

        ext_map = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg"}
        ext = ext_map.get(audio_fmt, ".wav")

        if output_format == "base64":
            return base64.b64encode(audio_data).decode("utf-8")
        return _save_audio(audio_data, ext)

    @staticmethod
    def _get_audio(status, api_key: str, client) -> tuple[bytes, str]:
        import httpx

        run_id = getattr(status, "run_id", None)
        if run_id:
            base = getattr(client, "_client_wrapper", None)
            if base and hasattr(base, "base_url"):
                url = f"{base.base_url}/tts-result/{run_id}"
            else:
                url = f"https://client.camb.ai/apis/tts-result/{run_id}"
            with httpx.Client() as http:
                resp = http.get(url, headers={"x-api-key": api_key})
                if resp.status_code == 200:
                    fmt = _detect_audio_format(resp.content, resp.headers.get("content-type", ""))
                    return resp.content, fmt

        message = getattr(status, "message", None)
        if message:
            msg_url = None
            if isinstance(message, dict):
                msg_url = message.get("output_url") or message.get("audio_url") or message.get("url")
            elif isinstance(message, str) and message.startswith("http"):
                msg_url = message
            if msg_url:
                with httpx.Client() as http:
                    resp = http.get(msg_url)
                    fmt = _detect_audio_format(resp.content, resp.headers.get("content-type", ""))
                    return resp.content, fmt

        raise RuntimeError("Failed to retrieve audio from translated TTS result")


class CambVoiceCloneTool(BaseTool):
    """Clone a voice from an audio sample using CAMB AI."""

    name: str = "camb_voice_clone"
    description: str = (
        "Clone a voice from an audio sample using CAMB AI. "
        "Requires 2+ seconds of audio. Returns the new voice ID."
    )
    args_schema: type[BaseModel] = VoiceCloneInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        voice_name: str,
        audio_file_path: str,
        gender: int,
        description: Optional[str] = None,
        age: Optional[int] = None,
        language: Optional[int] = None,
    ) -> str:
        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)

        with open(audio_file_path, "rb") as f:
            kwargs: dict[str, Any] = {"voice_name": voice_name, "gender": gender, "file": f}
            if description:
                kwargs["description"] = description
            if age:
                kwargs["age"] = age
            if language:
                kwargs["language"] = language
            result = client.voice_cloning.create_custom_voice(**kwargs)

        out = {
            "voice_id": getattr(result, "voice_id", getattr(result, "id", None)),
            "voice_name": voice_name,
            "status": "created",
        }
        if hasattr(result, "message"):
            out["message"] = result.message
        return json.dumps(out, indent=2)


class CambVoiceListTool(BaseTool):
    """List all available voices from CAMB AI."""

    name: str = "camb_voice_list"
    description: str = (
        "List all available voices from CAMB AI. "
        "Returns voice IDs, names, genders, ages, and languages. "
        "Use this to find the right voice_id for TTS tools."
    )
    args_schema: type[BaseModel] = VoiceListInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    GENDER_MAP: ClassVar[dict[int, str]] = {0: "not_specified", 1: "male", 2: "female", 9: "not_applicable"}

    def _run(self) -> str:
        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)
        voices = client.voice_cloning.list_voices()

        voice_list = []
        for v in voices:
            if isinstance(v, dict):
                voice_list.append({
                    "id": v.get("id"),
                    "name": v.get("voice_name", v.get("name", "Unknown")),
                    "gender": self.GENDER_MAP.get(v.get("gender", 0), "unknown"),
                    "age": v.get("age"),
                    "language": v.get("language"),
                })
            else:
                voice_list.append({
                    "id": getattr(v, "id", None),
                    "name": getattr(v, "voice_name", getattr(v, "name", "Unknown")),
                    "gender": self.GENDER_MAP.get(getattr(v, "gender", 0), "unknown"),
                    "age": getattr(v, "age", None),
                    "language": getattr(v, "language", None),
                })
        return json.dumps(voice_list, indent=2)


class CambVoiceFromDescriptionTool(BaseTool):
    """Generate a synthetic voice from a text description using CAMB AI."""

    name: str = "camb_voice_from_description"
    description: str = (
        "Generate a synthetic voice from a detailed text description using CAMB AI. "
        "Provide sample text and a voice description (accent, tone, age, gender, etc.). "
        "Returns preview audio URLs for the generated voice."
    )
    args_schema: type[BaseModel] = VoiceFromDescriptionInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        text: str,
        voice_description: str,
    ) -> str:
        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)

        result = client.text_to_voice.create_text_to_voice(
            text=text, voice_description=voice_description,
        )
        task_id = result.task_id
        status = _poll_task(client.text_to_voice.get_text_to_voice_status, task_id)
        run_id = status.run_id
        voice_result = client.text_to_voice.get_text_to_voice_result(run_id)

        out = {
            "previews": getattr(voice_result, "previews", []),
            "status": "completed",
        }
        return json.dumps(out, indent=2)


class CambTextToSoundTool(BaseTool):
    """Generate sounds, music, or soundscapes from text descriptions using CAMB AI."""

    name: str = "camb_text_to_sound"
    description: str = (
        "Generate sounds, music, or soundscapes from text descriptions using CAMB AI. "
        "Describe the audio you want and optionally specify duration and type."
    )
    args_schema: type[BaseModel] = TextToSoundInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        prompt: str,
        duration: Optional[float] = None,
        audio_type: Optional[str] = None,
        output_format: str = "file_path",
    ) -> str:
        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)

        kwargs: dict[str, Any] = {"prompt": prompt}
        if duration:
            kwargs["duration"] = duration
        if audio_type:
            kwargs["audio_type"] = audio_type

        result = client.text_to_audio.create_text_to_audio(**kwargs)
        task_id = result.task_id
        status = _poll_task(client.text_to_audio.get_text_to_audio_status, task_id)
        run_id = status.run_id

        audio_chunks: list[bytes] = []
        for chunk in client.text_to_audio.get_text_to_audio_result(run_id):
            audio_chunks.append(chunk)
        audio_data = b"".join(audio_chunks)

        if output_format == "base64":
            return base64.b64encode(audio_data).decode("utf-8")
        return _save_audio(audio_data, ".wav")


class CambAudioSeparationTool(BaseTool):
    """Separate vocals/speech from background audio using CAMB AI."""

    name: str = "camb_audio_separation"
    description: str = (
        "Separate vocals/speech from background audio using CAMB AI. "
        "Provide an audio URL or file path. "
        "Returns separate files for vocals and background audio."
    )
    args_schema: type[BaseModel] = AudioSeparationInput
    env_vars: list[EnvVar] = Field(default_factory=lambda: list(_COMMON_ENV_VARS))

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        audio_url: Optional[str] = None,
        audio_file_path: Optional[str] = None,
    ) -> str:
        key = self.api_key or os.environ.get("CAMB_API_KEY", "")
        client = _get_camb_client(key, self.base_url, self.timeout)

        kwargs: dict[str, Any] = {}
        if audio_url:
            import httpx

            resp = httpx.get(audio_url)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            try:
                with open(tmp_path, "rb") as f:
                    kwargs["media_file"] = f
                    result = client.audio_separation.create_audio_separation(**kwargs)
            finally:
                os.unlink(tmp_path)
        elif audio_file_path:
            with open(audio_file_path, "rb") as f:
                kwargs["media_file"] = f
                result = client.audio_separation.create_audio_separation(**kwargs)
        else:
            return json.dumps({"error": "No audio source provided"})

        task_id = result.task_id
        status = _poll_task(client.audio_separation.get_audio_separation_status, task_id)
        run_id = status.run_id
        sep = client.audio_separation.get_audio_separation_run_info(run_id)

        out: dict[str, Any] = {"vocals": None, "background": None, "status": "completed"}

        for attr, out_key in [("vocals_url", "vocals"), ("vocals", "vocals"),
                               ("voice_url", "vocals"),
                               ("background_url", "background"), ("background", "background"),
                               ("instrumental_url", "background")]:
            val = getattr(sep, attr, None)
            if val and out[out_key] is None:
                if isinstance(val, bytes):
                    out[out_key] = _save_audio(val, f"_{out_key}.wav")
                else:
                    out[out_key] = val

        return json.dumps(out, indent=2)


# ---------------------------------------------------------------------------
# Toolkit (convenience factory)
# ---------------------------------------------------------------------------

class CambAIToolkit:
    """Toolkit that bundles all CAMB AI tools for crewAI agents.

    Example::

        from crewai_tools.tools.camb_ai_tool import CambAIToolkit

        toolkit = CambAIToolkit()
        tools = toolkit.get_tools()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        include_tts: bool = True,
        include_translation: bool = True,
        include_transcription: bool = True,
        include_translated_tts: bool = True,
        include_voice_clone: bool = True,
        include_voice_list: bool = True,
        include_voice_from_description: bool = True,
        include_text_to_sound: bool = True,
        include_audio_separation: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._flags = {
            "tts": include_tts,
            "translation": include_translation,
            "transcription": include_transcription,
            "translated_tts": include_translated_tts,
            "voice_clone": include_voice_clone,
            "voice_list": include_voice_list,
            "voice_from_description": include_voice_from_description,
            "text_to_sound": include_text_to_sound,
            "audio_separation": include_audio_separation,
        }

    def get_tools(self) -> list[BaseTool]:
        key = self.api_key or os.environ.get("CAMB_API_KEY")
        if not key:
            raise ValueError(
                "CAMB AI API key is required. "
                "Set via 'api_key' parameter or CAMB_API_KEY environment variable."
            )
        common = {"api_key": key, "base_url": self.base_url, "timeout": self.timeout}
        mapping: list[tuple[str, type[BaseTool]]] = [
            ("tts", CambTTSTool),
            ("translation", CambTranslationTool),
            ("transcription", CambTranscriptionTool),
            ("translated_tts", CambTranslatedTTSTool),
            ("voice_clone", CambVoiceCloneTool),
            ("voice_list", CambVoiceListTool),
            ("voice_from_description", CambVoiceFromDescriptionTool),
            ("text_to_sound", CambTextToSoundTool),
            ("audio_separation", CambAudioSeparationTool),
        ]
        return [cls(**common) for flag, cls in mapping if self._flags.get(flag, True)]
