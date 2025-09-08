"""
Optimized Audio Segmenter with better performance for large files
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple
import logging
import time
from scipy.ndimage import uniform_filter1d
from .config import Config

logger = logging.getLogger(__name__)

class OptimizedAudioSegmenter:
    """
    Optimized version of AudioSegmenter with better performance for large files
    """
    
    def __init__(self, 
                 min_duration: float = 10.0,
                 max_duration: float = 15.0,
                 sample_rate: int = 24000,
                 overlap: float = 0.2):
        
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.overlap = overlap
        
    def simple_segment_audio(self, audio_path: Path, output_dir: Path, max_segments: int = 200) -> List[Path]:
        """
        Simple and fast audio segmentation with limits.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save segments  
            max_segments: Maximum number of segments to create
            
        Returns:
            List of paths to segmented audio files
        """
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Starting optimized segmentation of {audio_path.name}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        total_duration = len(audio) / sr
        logger.info(f"📊 Loaded audio: {total_duration:.2f}s at {sr}Hz")
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        # Calculate optimal segment size
        target_duration = (self.min_duration + self.max_duration) / 2  # 12.5s average
        estimated_segments = int(total_duration / target_duration)
        
        if estimated_segments > max_segments:
            logger.warning(f"⚠️ Would create {estimated_segments} segments, limiting to {max_segments}")
            target_duration = total_duration / max_segments
            if target_duration < self.min_duration:
                target_duration = self.min_duration
                max_segments = int(total_duration / self.min_duration)
                logger.info(f"📝 Adjusted to {max_segments} segments of {target_duration:.1f}s each")
        
        # Simple segmentation with VAD-based boundaries
        segments = self.create_segments_with_boundaries(audio, sr, target_duration, max_segments)
        
        # Save segments
        segment_paths = []
        logger.info(f"💾 Saving {len(segments)} segments...")
        
        for i, (start_sample, end_sample) in enumerate(segments):
            segment_audio = audio[start_sample:end_sample]
            duration = len(segment_audio) / sr
            
            if duration >= self.min_duration:
                filename = f"{audio_path.stem}_segment_{i:04d}.{Config.AUDIO_FORMAT}"
                segment_path = output_dir / filename
                
                sf.write(segment_path, segment_audio, sr)
                segment_paths.append(segment_path)
                
                if i % 50 == 0:  # Log progress every 50 segments
                    logger.info(f"📦 Saved {i+1}/{len(segments)} segments...")
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Created {len(segment_paths)} segments in {processing_time:.2f}s")
        
        return segment_paths
    
    def create_segments_with_boundaries(self, audio: np.ndarray, sr: int, 
                                      target_duration: float, max_segments: int) -> List[Tuple[int, int]]:
        """Create segments with speech boundary detection."""
        
        # Quick silence detection for boundaries
        silence_mask = self.detect_silence_fast(audio, sr)
        
        # Find potential cut points
        potential_cuts = self.find_silence_boundaries(silence_mask, sr)
        
        # Create segments with target duration
        segments = []
        current_pos = 0
        segment_samples = int(target_duration * sr)
        total_samples = len(audio)
        
        for i in range(max_segments):
            if current_pos >= total_samples:
                break
                
            # Target end position
            target_end = min(current_pos + segment_samples, total_samples)
            
            # Find best cut point near target
            best_cut = self.find_best_cut_near_target(potential_cuts, target_end, sr)
            if best_cut is None or best_cut <= current_pos:
                best_cut = target_end
            
            # Ensure minimum duration
            if (best_cut - current_pos) / sr < self.min_duration and i < max_segments - 1:
                # Extend to minimum duration
                best_cut = min(current_pos + int(self.min_duration * sr), total_samples)
            
            segments.append((current_pos, best_cut))
            current_pos = best_cut
            
            if current_pos >= total_samples:
                break
        
        return segments
    
    def detect_silence_fast(self, audio: np.ndarray, sr: int, 
                           frame_length: int = 2048, threshold_db: float = -40) -> np.ndarray:
        """Fast silence detection."""
        
        # Use librosa's built-in RMS for speed
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=frame_length//4)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms)
        
        # Smooth slightly
        if len(rms_db) > 10:
            rms_db = uniform_filter1d(rms_db, size=5)
        
        return rms_db < threshold_db
    
    def find_silence_boundaries(self, silence_mask: np.ndarray, sr: int, 
                              min_silence_duration: float = 0.3) -> List[int]:
        """Find silence boundaries suitable for cutting."""
        
        hop_length = sr // 4  # 4 times per second
        min_silence_frames = int(min_silence_duration * 4)  # frames needed for min silence
        
        boundaries = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                silence_start = i
                in_silence = True
            elif not is_silent and in_silence:
                silence_duration = i - silence_start
                if silence_duration >= min_silence_frames:
                    # Add middle of silence region as boundary
                    boundary_frame = (silence_start + i) // 2
                    boundary_sample = boundary_frame * hop_length
                    boundaries.append(boundary_sample)
                in_silence = False
        
        return sorted(boundaries)
    
    def find_best_cut_near_target(self, boundaries: List[int], target: int, 
                                sr: int, search_window: float = 2.0) -> int:
        """Find best cut point near target position."""
        
        if not boundaries:
            return target
            
        window_samples = int(search_window * sr)
        candidates = [b for b in boundaries if abs(b - target) <= window_samples]
        
        if not candidates:
            return target
            
        # Return closest to target
        return min(candidates, key=lambda x: abs(x - target))
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.9  # Leave some headroom
        return audio
        

# Monkey patch the original segmenter
def patch_audio_segmenter():
    """Replace the original segment_audio method with optimized version."""
    from . import audio_segmenter
    
    original_class = audio_segmenter.AudioSegmenter
    optimized_segmenter = OptimizedAudioSegmenter()
    
    def optimized_segment_audio(self, audio_path: Path, output_dir: Path) -> List[Path]:
        logger.info("🔄 Using optimized segmentation algorithm")
        return optimized_segmenter.simple_segment_audio(audio_path, output_dir)
    
    # Replace the method
    original_class.segment_audio = optimized_segment_audio
    
    logger.info("✅ Audio segmenter patched with optimized version")


if __name__ == "__main__":
    # Test the optimized segmenter
    logging.basicConfig(level=logging.INFO)
    
    segmenter = OptimizedAudioSegmenter()
    # segments = segmenter.simple_segment_audio(Path("test.flac"), Path("segments/"))
    # print(f"Created {len(segments)} segments")
