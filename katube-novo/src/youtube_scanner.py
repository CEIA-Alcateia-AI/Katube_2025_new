"""
YouTube Channel Scanner
Integrates with the original katube search.py for channel/playlist scanning
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
import re

# Add buscador to path for imports
BUSCADOR_PATH = Path(__file__).resolve().parent.parent / "buscador"
sys.path.insert(0, str(BUSCADOR_PATH))

'''
try:
    from search import search_videos, get_videos
    from googleapiclient.discovery import build
    SEARCH_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import search module: {e}")
    search_videos = None
    get_videos = None
    build = None
    SEARCH_MODULES_AVAILABLE = False'''

logger = logging.getLogger(__name__)

class YouTubeChannelScanner:
    """
    YouTube channel/playlist scanner using the original katube search.py
    """
    
    def __init__(self, api_key: str, base_dir: Path = None):
        """
        Initialize YouTube scanner.
        
        Args:
            api_key: YouTube Data API key
            base_dir: Base directory for storing results
        """
        self.api_key = api_key
        self.base_dir = base_dir or Path.cwd() / "youtube_scans"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if not api_key:
            logger.warning("⚠️ No YouTube API key provided. Channel scanning will not work.")
    
    def extract_channel_id(self, url: str) -> Optional[str]:
        """
        Extract channel ID from various YouTube URL formats.
        
        Args:
            url: YouTube channel URL
            
        Returns:
            Channel ID or None if not found
        """
        try:
            # Handle different URL formats
            patterns = [
                r'youtube\.com/channel/([a-zA-Z0-9_-]+)',
                r'youtube\.com/c/([a-zA-Z0-9_-]+)',
                r'youtube\.com/@([a-zA-Z0-9_-]+)',
                r'youtube\.com/user/([a-zA-Z0-9_-]+)',
                r'youtube\.com/([a-zA-Z0-9_-]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    channel_identifier = match.group(1)
                    
                    # If it's a channel ID (starts with UC), return directly
                    if channel_identifier.startswith('UC'):
                        return channel_identifier
                    
                    # Otherwise, resolve to channel ID
                    return self._resolve_channel_id(channel_identifier)
            
            logger.warning(f"⚠️ Could not extract channel ID from URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting channel ID: {e}")
            return None
    
    def _resolve_channel_id(self, identifier: str) -> Optional[str]:
        """
        Resolve channel identifier to actual channel ID.
        
        Args:
            identifier: Channel handle, username, or custom URL
            
        Returns:
            Channel ID or None
        """
        try:
            if not self.api_key or not build:
                logger.warning("⚠️ YouTube API not available for channel resolution")
                return None
            
            youtube = build('youtube', 'v3', developerKey=self.api_key)
            
            # Try different methods to resolve channel ID
            methods = [
                # Try by handle (@username)
                lambda: youtube.channels().list(part='id', forHandle=identifier).execute(),
                # Try by username
                lambda: youtube.channels().list(part='id', forUsername=identifier).execute(),
                # Try as custom URL
                lambda: youtube.channels().list(part='id', forUsername=identifier).execute(),
            ]
            
            for method in methods:
                try:
                    response = method()
                    if response.get('items'):
                        return response['items'][0]['id']
                except:
                    continue
            
            logger.warning(f"⚠️ Could not resolve channel ID for: {identifier}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error resolving channel ID: {e}")
            return None
    
    def scan_channel(self, channel_url: str, output_filename: str = "youtube_videos.txt") -> Optional[Path]:
        """
        Scan YouTube channel for all videos.
        
        Args:
            channel_url: YouTube channel URL
            output_filename: Output filename for video list
            
        Returns:
            Path to output file with video URLs or None if failed
        """
        try:
            if not self.api_key or not search_videos:
                logger.error("❌ YouTube API key or search module not available")
                return None
            
            # Extract channel ID
            channel_id = self.extract_channel_id(channel_url)
            if not channel_id:
                logger.error(f"❌ Could not extract channel ID from: {channel_url}")
                return None
            
            logger.info(f"🔍 Scanning channel: {channel_id}")
            
            # Create output directory
            channel_dir = self.base_dir / f"channel_{channel_id}"
            channel_dir.mkdir(parents=True, exist_ok=True)
            
            # Use original search_videos function
            output_path = search_videos(
                api_key=self.api_key,
                content_id=channel_id,
                output_folderpath=str(self.base_dir),
                output_result_file=output_filename
            )
            
            if output_path and Path(output_path).exists():
                logger.info(f"✅ Channel scan complete: {output_path}")
                return Path(output_path)
            else:
                logger.error("❌ Channel scan failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error scanning channel: {e}")
            return None
    
    def get_video_urls(self, video_list_path: Path) -> List[str]:
        """
        Read video URLs from file.
        
        Args:
            video_list_path: Path to file containing video URLs
            
        Returns:
            List of video URLs
        """
        try:
            if not video_list_path.exists():
                logger.error(f"❌ Video list file not found: {video_list_path}")
                return []
            
            with open(video_list_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            logger.info(f"📋 Found {len(urls)} video URLs")
            return urls
            
        except Exception as e:
            logger.error(f"❌ Error reading video URLs: {e}")
            return []
    
    def scan_and_process_channel(self, channel_url: str, process_callback=None) -> Dict[str, Any]:
        """
        Scan channel and optionally process each video.
        
        Args:
            channel_url: YouTube channel URL
            process_callback: Optional callback function to process each video URL (video_url, total_videos, current_index)
            
        Returns:
            Dictionary with scan results
        """
        try:
            # Scan channel
            video_list_path = self.scan_channel(channel_url)
            if not video_list_path:
                return {
                    'success': False,
                    'error': 'Channel scan failed',
                    'videos_processed': 0,
                    'videos_failed': 0
                }
            
            # Get video URLs
            video_urls = self.get_video_urls(video_list_path)
            if not video_urls:
                return {
                    'success': False,
                    'error': 'No videos found',
                    'videos_processed': 0,
                    'videos_failed': 0
                }
            
            # Process videos if callback provided
            processed_count = 0
            failed_count = 0
            
            if process_callback:
                logger.info(f"🔄 Processing {len(video_urls)} videos...")
                
                for i, video_url in enumerate(video_urls):
                    try:
                        logger.info(f"📹 Processing video {i+1}/{len(video_urls)}: {video_url}")
                        
                        # Call the processing callback with total and current index
                        result = process_callback(video_url, len(video_urls), i+1)
                        
                        if result:
                            processed_count += 1
                            logger.info(f"✅ Video processed successfully")
                        else:
                            failed_count += 1
                            logger.warning(f"⚠️ Video processing failed")
                            
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"❌ Error processing video {video_url}: {e}")
            
            return {
                'success': True,
                'video_list_path': str(video_list_path),
                'total_videos': len(video_urls),
                'videos_processed': processed_count,
                'videos_failed': failed_count,
                'video_urls': video_urls
            }
            
        except Exception as e:
            logger.error(f"❌ Error in scan_and_process_channel: {e}")
            return {
                'success': False,
                'error': str(e),
                'videos_processed': 0,
                'videos_failed': 0
            }


if __name__ == "__main__":
    # Test the scanner
    logging.basicConfig(level=logging.INFO)
    
    # You need to set your YouTube API key
    api_key = os.getenv('YOUTUBE_API_KEY', '')
    
    if not api_key:
        print("❌ Please set YOUTUBE_API_KEY environment variable")
        sys.exit(1)
    
    scanner = YouTubeChannelScanner(api_key)
    
    # Test with a channel URL
    test_url = "https://www.youtube.com/@example"
    result = scanner.scan_channel(test_url)
    
    if result:
        print(f"✅ Scan successful: {result}")
        urls = scanner.get_video_urls(result)
        print(f"📋 Found {len(urls)} videos")
    else:
        print("❌ Scan failed")
