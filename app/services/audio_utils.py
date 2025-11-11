"""
audio_utils.py
--------------
Handles all audio preprocessing and format handling.
Keeps non-ML logic separated from model inference logic.
"""

import tempfile
from pydub import AudioSegment


def convert_to_wav(uploaded_file):
    """
    Converts uploaded audio file (any format) to WAV for Whisper.
    Returns: path to converted WAV file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        try:
            sound = AudioSegment.from_file(uploaded_file.file)
            sound.export(temp_audio.name, format="wav")
            return temp_audio.name
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed: {e}")


def get_audio_duration(wav_path: str) -> float:
    """
    Returns the duration of an audio file in seconds.
    Useful for analytics or logging.
    """
    try:
        sound = AudioSegment.from_wav(wav_path)
        return round(len(sound) / 1000, 2)
    except Exception as e:
        raise RuntimeError(f"Unable to calculate audio duration: {e}")
