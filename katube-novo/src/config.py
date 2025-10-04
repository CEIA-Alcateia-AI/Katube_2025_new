"""
Configuration settings for the audio processing pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Audio settings
    AUDIO_FORMAT = os.getenv('AUDIO_FORMAT', 'flac')
    AUDIO_QUALITY = os.getenv('AUDIO_QUALITY', 'best')
    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '48000'))
    SEGMENT_MIN_DURATION = float(os.getenv('SEGMENT_MIN_DURATION', '10.0'))
    SEGMENT_MAX_DURATION = float(os.getenv('SEGMENT_MAX_DURATION', '15.0'))
    SEGMENT_OVERLAP = float(os.getenv('SEGMENT_OVERLAP', '0.5'))
    
    # Diarization settings
    PYANNOTE_MODEL = os.getenv('PYANNOTE_MODEL', 'pyannote/speaker-diarization-3.1')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    
    # Voice overlap detection
    OVERLAP_THRESHOLD = float(os.getenv('OVERLAP_THRESHOLD', '0.9'))  # Increased from 0.5 to 0.9 (90% overlap required to classify as overlapping)
    MIN_SPEECH_DURATION = float(os.getenv('MIN_SPEECH_DURATION', '0.5'))
    
    # Audio segmentation limits
    MAX_SEGMENTS = int(os.getenv('MAX_SEGMENTS', '5000'))  # Maximum segments per audio
    
    # MOS Quality Filter settings (OBRIGATÓRIO)
    MOS_THRESHOLD = float(os.getenv('MOS_THRESHOLD', '3.0'))  # Minimum MOS score to accept
    # ENABLE_MOS_FILTER sempre True - filtro é obrigatório
    
    # YouTube API settings
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    # Directories
    BASE_DIR = Path(__file__).parent.parent
    AUDIOS_BAIXADOS_DIR = Path(os.getenv('AUDIOS_BAIXADOS_DIR', r'C:\Igor\BIA\Alcateia\Katube_2025_new\audios_download'))
    OUTPUT_DIR = AUDIOS_BAIXADOS_DIR / "output"
    SEGMENTS_DIR = OUTPUT_DIR / "segments"
    SPEAKERS_DIR = OUTPUT_DIR / "speakers"
    TEMP_DIR = OUTPUT_DIR / "audio_bruto"

    # YouTube download settings
    YOUTUBE_FORMAT = "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best/worst"
    
    # STT preparation settings
    MAX_SEGMENT_SIZE = 25 * 1024 * 1024  # 25MB max per segment for STT
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for dir_path in [cls.OUTPUT_DIR, cls.TEMP_DIR, cls.SEGMENTS_DIR, cls.SPEAKERS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
