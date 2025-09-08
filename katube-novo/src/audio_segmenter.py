"""
Intelligent audio segmentation that preserves word boundaries and speech patterns.
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import webrtcvad
from scipy.signal import find_peaks
import pyloudnorm as pyln

from .config import Config

logger = logging.getLogger(__name__)

class AudioSegmenter:
    def __init__(self, min_duration: float = None, max_duration: float = None):
        self.min_duration = min_duration or Config.SEGMENT_MIN_DURATION
        self.max_duration = max_duration or Config.SEGMENT_MAX_DURATION
        self.sample_rate = Config.SAMPLE_RATE
        self.overlap = Config.SEGMENT_OVERLAP
        
        # VAD for speech detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level (0-3)
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio using loudness normalization."""
        try:
            meter = pyln.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(audio)
            # Normalize to -23 LUFS (broadcast standard)
            audio_normalized = pyln.normalize.loudness(audio, loudness, -23.0)
            return audio_normalized
        except:
            # Fallback to simple normalization
            return audio / np.max(np.abs(audio))
    
    def detect_speech_activity(self, audio: np.ndarray, frame_duration: int = 30) -> List[bool]:
        """Detect speech activity using WebRTC VAD."""
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16)
        
        frame_size = int(self.sample_rate * frame_duration / 1000)  # frame_duration in ms
        frames = []
        
        for i in range(0, len(audio_16bit), frame_size):
            frame = audio_16bit[i:i+frame_size]
            if len(frame) == frame_size:
                frames.append(frame.tobytes())
        
        # Apply VAD
        speech_frames = []
        for frame in frames:
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                speech_frames.append(is_speech)
            except:
                speech_frames.append(False)
        
        return speech_frames
    
    def detect_silence(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Detect silence regions in audio."""
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio))
        
        # Smooth the signal  
        hop_length = 512
        # Use scipy.ndimage.uniform_filter1d instead of deprecated smooth
        from scipy.ndimage import uniform_filter1d
        audio_db = uniform_filter1d(audio_db, size=hop_length//4)
        
        # Find silence (below threshold)
        silence_mask = audio_db < threshold_db
        
        return silence_mask
    
    def find_optimal_cut_points(self, audio: np.ndarray) -> List[int]:
        """Find optimal points to cut audio based on silence and speech patterns."""
        silence_mask = self.detect_silence(audio)
        
        # Find silence regions
        silence_starts = []
        silence_ends = []
        in_silence = False
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                silence_starts.append(i)
                in_silence = True
            elif not is_silent and in_silence:
                silence_ends.append(i)
                in_silence = False
        
        # Ensure we have matching starts and ends
        if in_silence:
            silence_ends.append(len(silence_mask))
        
        # Find silence regions longer than 0.3 seconds
        min_silence_samples = int(0.3 * self.sample_rate / 512)  # hop_length = 512
        
        good_cut_points = []
        for start, end in zip(silence_starts, silence_ends):
            if end - start > min_silence_samples:
                # Use middle of silence region as cut point
                cut_point = (start + end) // 2 * 512  # Convert back to sample index
                good_cut_points.append(cut_point)
        
        return sorted(good_cut_points)
    
    def segment_audio(self, audio_path: Path, output_dir: Path) -> List[Path]:
        """
        Segment audio file intelligently based on speech patterns.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save segments
            
        Returns:
            List of paths to segmented audio files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        logger.info(f"Loaded audio: {len(audio) / sr:.2f}s at {sr}Hz")
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        # Find optimal cut points
        cut_points = self.find_optimal_cut_points(audio)
        
        # Add start and end points
        cut_points = [0] + cut_points + [len(audio)]
        cut_points = sorted(list(set(cut_points)))  # Remove duplicates
        
        logger.info(f"Found {len(cut_points) - 1} potential segments")
        
        # Create segments with duration constraints
        segments = []
        segment_paths = []
        
        i = 0
        segment_idx = 0
        
        while i < len(cut_points) - 1:
            start_sample = cut_points[i]
            end_sample = cut_points[i + 1]
            
            # Calculate duration
            duration = (end_sample - start_sample) / sr
            
            # If segment is too short, merge with next
            if duration < self.min_duration and i < len(cut_points) - 2:
                continue
            
            # If segment is too long, split it
            if duration > self.max_duration:
                # Split into smaller segments at silence points within the range
                sub_cuts = [cp for cp in cut_points if start_sample < cp < end_sample]
                
                if sub_cuts:
                    # Use the cut point closest to max_duration
                    target_sample = start_sample + int(self.max_duration * sr)
                    best_cut = min(sub_cuts, key=lambda x: abs(x - target_sample))
                    end_sample = best_cut
                else:
                    # Force split at max_duration
                    end_sample = start_sample + int(self.max_duration * sr)
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            duration = len(segment_audio) / sr
            
            # Save segment
            if duration >= self.min_duration:
                filename = f"{audio_path.stem}_segment_{segment_idx:04d}.{Config.AUDIO_FORMAT}"
                segment_path = output_dir / filename
                
                sf.write(segment_path, segment_audio, sr)
                segment_paths.append(segment_path)
                
                logger.debug(f"Segment {segment_idx}: {duration:.2f}s -> {segment_path}")
                segment_idx += 1
            
            # Move to next segment
            # Add overlap if needed
            overlap_samples = int(self.overlap * sr)
            next_start = max(start_sample + 1, end_sample - overlap_samples)
            
            # Find next cut point after the overlap
            i += 1
            while i < len(cut_points) and cut_points[i] <= next_start:
                i += 1
            
            if i >= len(cut_points):
                break
        
        logger.info(f"Created {len(segment_paths)} segments")
        return segment_paths
    
    def segment_with_timestamps(self, audio_path: Path, output_dir: Path) -> List[Tuple[Path, float, float]]:
        """
        Segment audio and return with timestamps.
        
        Returns:
            List of (path, start_time, end_time) tuples
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        # Find optimal cut points
        cut_points = self.find_optimal_cut_points(audio)
        cut_points = [0] + cut_points + [len(audio)]
        cut_points = sorted(list(set(cut_points)))
        
        segments_with_timestamps = []
        i = 0
        segment_idx = 0
        
        while i < len(cut_points) - 1:
            start_sample = cut_points[i]
            end_sample = cut_points[i + 1]
            duration = (end_sample - start_sample) / sr
            
            if duration < self.min_duration and i < len(cut_points) - 2:
                continue
                
            if duration > self.max_duration:
                sub_cuts = [cp for cp in cut_points if start_sample < cp < end_sample]
                if sub_cuts:
                    target_sample = start_sample + int(self.max_duration * sr)
                    best_cut = min(sub_cuts, key=lambda x: abs(x - target_sample))
                    end_sample = best_cut
                else:
                    end_sample = start_sample + int(self.max_duration * sr)
            
            segment_audio = audio[start_sample:end_sample]
            duration = len(segment_audio) / sr
            
            if duration >= self.min_duration:
                filename = f"{audio_path.stem}_segment_{segment_idx:04d}.{Config.AUDIO_FORMAT}"
                segment_path = output_dir / filename
                
                sf.write(segment_path, segment_audio, sr)
                
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                segments_with_timestamps.append((segment_path, start_time, end_time))
                segment_idx += 1
            
            # Move to next
            overlap_samples = int(self.overlap * sr)
            next_start = max(start_sample + 1, end_sample - overlap_samples)
            
            i += 1
            while i < len(cut_points) and cut_points[i] <= next_start:
                i += 1
                
            if i >= len(cut_points):
                break
        
        return segments_with_timestamps


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    segmenter = AudioSegmenter()
    # segments = segmenter.segment_audio(Path("input.flac"), Path("segments/"))
    # print(f"Created {len(segments)} segments")
