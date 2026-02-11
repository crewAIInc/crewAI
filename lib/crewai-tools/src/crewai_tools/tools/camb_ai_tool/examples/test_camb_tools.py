"""Example usage of CAMB AI tools with crewAI."""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parents[7] / ".env")

from crewai_tools.tools.camb_ai_tool import (
    CambAIToolkit,
    CambAudioSeparationTool,
    CambTextToSoundTool,
    CambTranscriptionTool,
    CambTranslatedTTSTool,
    CambTranslationTool,
    CambTTSTool,
    CambVoiceCloneTool,
    CambVoiceFromDescriptionTool,
    CambVoiceListTool,
)

API_KEY = os.environ.get("CAMB_API_KEY")
if not API_KEY:
    raise RuntimeError("Set CAMB_API_KEY environment variable to run examples")

AUDIO_SAMPLE = os.environ.get(
    "CAMB_AUDIO_SAMPLE",
    str(Path(__file__).resolve().parents[8] / "yt-dlp" / "voices" / "original" / "sabrina-original-clip.mp3"),
)


def play(path: str):
    """Play an audio file with afplay (macOS)."""
    if sys.platform == "darwin":
        print(f"  Playing: {path}")
        subprocess.run(["afplay", path], check=False)
    else:
        print(f"  Audio file at: {path} (afplay not available on this platform)")


def test_tts():
    """1. Text-to-Speech: convert text to audio."""
    tool = CambTTSTool(api_key=API_KEY)
    path = tool._run(text="Hello from CAMB AI and crewAI! This is a text to speech test.", language="en-us")
    print(f"  Audio saved to: {path}")
    assert path.endswith(".wav")
    play(path)


def test_translation():
    """2. Translation: translate text between languages."""
    tool = CambTranslationTool(api_key=API_KEY)
    result = tool._run(text="Hello, how are you?", source_language=1, target_language=2)
    print(f"  Result: {result}")
    assert len(result) > 0


def test_voice_list():
    """3. Voice List: list available voices."""
    tool = CambVoiceListTool(api_key=API_KEY)
    result = tool._run()
    print(f"  Voices (first 200 chars): {result[:200]}")
    assert "id" in result


def test_transcription():
    """4. Transcription: transcribe audio from local file."""
    tool = CambTranscriptionTool(api_key=API_KEY)
    result = tool._run(language=1, audio_file_path=AUDIO_SAMPLE)
    print(f"  Transcription (first 300 chars): {result[:300]}")
    assert "text" in result


def test_translated_tts():
    """5. Translated TTS: translate and speak in one step."""
    tool = CambTranslatedTTSTool(api_key=API_KEY)
    path = tool._run(text="Hello, how are you?", source_language=1, target_language=2)
    print(f"  Audio saved to: {path}")
    assert path.endswith((".wav", ".mp3", ".flac", ".ogg"))
    play(path)


def test_text_to_sound():
    """6. Text-to-Sound: generate audio from a description."""
    tool = CambTextToSoundTool(api_key=API_KEY)
    path = tool._run(prompt="gentle rain on a rooftop", duration=5.0, audio_type="sound")
    print(f"  Audio saved to: {path}")
    assert path.endswith(".wav")
    play(path)


def test_voice_clone():
    """7. Voice Clone: clone a voice from an audio sample."""
    tool = CambVoiceCloneTool(api_key=API_KEY)
    result = tool._run(voice_name="test_clone_crewai", audio_file_path=AUDIO_SAMPLE, gender=2)
    print(f"  Result: {result}")
    assert "voice_id" in result


def test_audio_separation():
    """8. Audio Separation: separate vocals from background."""
    tool = CambAudioSeparationTool(api_key=API_KEY)
    result = tool._run(audio_file_path=AUDIO_SAMPLE)
    print(f"  Result: {result}")
    assert "status" in result


def test_voice_from_description():
    """9. Voice from Description: generate a voice from text description."""
    tool = CambVoiceFromDescriptionTool(api_key=API_KEY)
    result = tool._run(
        text="Hello, this is a comprehensive test of the voice generation feature from CAMB AI. We are testing whether we can create a new synthetic voice from just a text description alone.",
        voice_description="A warm, friendly female voice with a slight British accent, aged around 30, professional tone suitable for narration and audiobooks, clear enunciation with a calm demeanor",
    )
    print(f"  Result (first 200 chars): {result[:200]}")
    assert "previews" in result


if __name__ == "__main__":
    tests = [
        test_tts,
        test_translation,
        test_voice_list,
        test_transcription,
        test_translated_tts,
        test_text_to_sound,
        test_voice_clone,
        test_audio_separation,
        test_voice_from_description,
    ]
    for t in tests:
        print(f"\n--- {t.__doc__} ---")
        try:
            t()
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
