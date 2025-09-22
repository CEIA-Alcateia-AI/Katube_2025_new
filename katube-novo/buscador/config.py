"""
Configuration for YouTube channel scanner
Compatible with original katube search.py
"""
import os

class Config:
    """Configuration for YouTube scanning."""
    
    # YouTube API configuration
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')
    
    # Base configuration for katube compatibility
    orig_base = 'channel'  # Default to channel scanning
    
    # Output configuration
    OUTPUT_DIR = 'youtube_scans'
    OUTPUT_FILENAME = 'youtube_videos.txt'
    
    # Processing limits
    MAX_VIDEOS_PER_CHANNEL = 1000  # Limit to prevent API quota issues
    MAX_RESULTS_PER_REQUEST = 50    # YouTube API limit
