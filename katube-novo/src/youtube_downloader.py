"""
YouTube audio downloader with highest quality FLAC output.
"""
import os
import yt_dlp
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import subprocess
import sys

from .config import Config

logger = logging.getLogger(__name__)

class YouTubeDownloader:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure UTF-8 encoding for subprocesses
        self._setup_encoding()
    
    def _setup_encoding(self):
        """Configure UTF-8 encoding for subprocesses to handle special characters."""
        if sys.platform == "win32":
            # Set environment variables for UTF-8 on Windows
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.environ['PYTHONUTF8'] = '1'
            os.environ['LANG'] = 'en_US.UTF-8'
            os.environ['LC_ALL'] = 'en_US.UTF-8'
            
            # Configure subprocess to use UTF-8
            if hasattr(subprocess, '_default_encoding'):
                subprocess._default_encoding = 'utf-8'
            if hasattr(subprocess, '_default_errors'):
                subprocess._default_errors = 'replace'
            
            # Set console code page to UTF-8
            try:
                subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            except:
                pass
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to handle UTF-8 characters safely."""
        import unicodedata
        import re
        
        # Normalize Unicode characters
        filename = unicodedata.normalize('NFKD', filename)
        
        # Replace problematic characters
        filename = re.sub(r'[^\w\s\-_\.]', '_', filename)
        
        # Replace multiple spaces/underscores with single underscore
        filename = re.sub(r'[\s_]+', '_', filename)
        
        # Remove leading/trailing underscores
        filename = filename.strip('_')
        
        return filename
        
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
            # UTF-8 encoding options
            'encoding': 'utf-8',
            'no_warnings': True,
            'quiet': True,
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
            # Get video info first with UTF-8 encoding
            ydl_opts_info = {
                'quiet': True,
                'encoding': 'utf-8',
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'unknown')
                video_id = info.get('id', 'unknown')
                duration = info.get('duration', 0)
                
            logger.info(f"Video: {video_title} (Duration: {duration}s)")
            
            # Set output filename
            if custom_filename:
                filename = custom_filename
            else:
                # Clean filename and handle UTF-8 characters
                filename = self._sanitize_filename(f"{video_id}")
                filename = filename[:100]  # Limit length
                
            output_path = str(self.output_dir / f"{filename}.%(ext)s")
            
            # Download with options
            ydl_opts = self._get_ydl_opts(output_path)
            
            # Configure environment for UTF-8
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            env['LANG'] = 'en_US.UTF-8'
            env['LC_ALL'] = 'en_US.UTF-8'
            
            # Add encoding to ydl_opts
            ydl_opts['encoding'] = 'utf-8'
            ydl_opts['env'] = env
            
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