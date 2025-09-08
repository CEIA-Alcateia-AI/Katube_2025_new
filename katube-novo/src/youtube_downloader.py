"""
YouTube audio downloader with highest quality FLAC output.
"""
import os
import yt_dlp
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .config import Config

logger = logging.getLogger(__name__)

class YouTubeDownloader:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Config.TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_ydl_opts(self, output_path: str) -> Dict[str, Any]:
        """Get yt-dlp options for highest quality audio download."""
        return {
            'format': Config.YOUTUBE_FORMAT,
            'outtmpl': output_path,
            'noplaylist': True,
            'extractaudio': True,
            'audioformat': Config.AUDIO_FORMAT,
            'audioquality': '0',  # Best quality
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': Config.AUDIO_FORMAT,
                'preferredquality': '0',  # Best quality
            }, {
                'key': 'FFmpegMetadata',
                'add_metadata': True,
            }],
            'postprocessor_args': [
                '-ar', str(Config.SAMPLE_RATE),  # Sample rate
                '-ac', '1',  # Mono
            ],
        }
    
    def download(self, url: str, custom_filename: Optional[str] = None) -> Path:
        """
        Download audio from YouTube URL.
        
        Args:
            url: YouTube URL
            custom_filename: Custom filename (without extension)
            
        Returns:
            Path to downloaded audio file
        """
        try:
            # Get video info first
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'unknown')
                video_id = info.get('id', 'unknown')
                duration = info.get('duration', 0)
                
            logger.info(f"Video: {video_title} (Duration: {duration}s)")
            
            # Set output filename
            if custom_filename:
                filename = custom_filename
            else:
                # Clean filename
                filename = "".join(c for c in f"{video_id}_{video_title}" if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = filename.replace(' ', '_')[:100]  # Limit length
                
            output_path = str(self.output_dir / f"{filename}.%(ext)s")
            
            # Download with options
            ydl_opts = self._get_ydl_opts(output_path)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            # Find the downloaded file
            expected_file = self.output_dir / f"{filename}.{Config.AUDIO_FORMAT}"
            if expected_file.exists():
                logger.info(f"Downloaded: {expected_file}")
                return expected_file
            else:
                # Search for file with similar name
                for file in self.output_dir.glob(f"{filename}.*"):
                    if file.suffix[1:] == Config.AUDIO_FORMAT:
                        logger.info(f"Found downloaded file: {file}")
                        return file
                        
                raise FileNotFoundError(f"Downloaded file not found: {expected_file}")
                
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            raise
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without downloading."""
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            return ydl.extract_info(url, download=False)


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    downloader = YouTubeDownloader()
    # url = "https://www.youtube.com/watch?v=example"
    # audio_file = downloader.download(url)
    # print(f"Downloaded: {audio_file}")
