"""
Voice overlap detection to identify overlapping speech segments.
"""
import librosa
import numpy as np
import soundfile as sf
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from scipy import signal
from scipy.stats import pearsonr
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from .config import Config

logger = logging.getLogger(__name__)

class OverlapDetector:
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.overlap_threshold = Config.OVERLAP_THRESHOLD
        self.min_speech_duration = Config.MIN_SPEECH_DURATION
        
        # Load pre-trained model for feature extraction
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Wav2Vec2 model for speech feature extraction."""
        try:
            model_name = "facebook/wav2vec2-base"
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            logger.info("Loaded Wav2Vec2 model for overlap detection")
        except Exception as e:
            logger.warning(f"Could not load Wav2Vec2 model: {e}")
            self.processor = None
            self.model = None
    
    def compute_energy(self, audio: np.ndarray, frame_size: int = 2048, hop_length: int = 512) -> np.ndarray:
        """Compute energy-based features."""
        # RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_size, hop_length=hop_length)[0]
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=hop_length)[0]
        
        # Combine features
        features = np.vstack([rms, zcr, spectral_centroid])
        return features
    
    def compute_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Compute MFCC features for overlap detection."""
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=13,
            hop_length=512
        )
        
        # Add delta and delta-delta features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Combine features
        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        return features
    
    def detect_multiple_speakers_energy(self, audio: np.ndarray, window_size: float = 0.5) -> List[Tuple[float, float, float]]:
        """
        Detect overlapping speech using energy-based analysis.
        
        Returns:
            List of (start_time, end_time, overlap_confidence) tuples
        """
        overlaps = []
        window_samples = int(window_size * self.sample_rate)
        hop_samples = window_samples // 2
        
        for i in range(0, len(audio) - window_samples, hop_samples):
            window = audio[i:i + window_samples]
            
            # Compute spectral features
            stft = librosa.stft(window, hop_length=256)
            magnitude = np.abs(stft)
            
            # Analyze spectral characteristics
            freq_bins = magnitude.shape[0]
            
            # Look for multiple energy peaks in frequency domain
            energy_per_freq = np.mean(magnitude, axis=1)
            peaks, _ = signal.find_peaks(energy_per_freq, height=np.max(energy_per_freq) * 0.3)
            
            # Check for harmonic structure indicating multiple speakers
            if len(peaks) > 2:
                # Analyze temporal variation
                temporal_var = np.var(magnitude, axis=1)
                high_var_freqs = np.sum(temporal_var > np.mean(temporal_var))
                
                # Calculate overlap confidence
                overlap_confidence = min(1.0, (len(peaks) * high_var_freqs) / (freq_bins * 0.5))
                
                if overlap_confidence > self.overlap_threshold:
                    start_time = i / self.sample_rate
                    end_time = (i + window_samples) / self.sample_rate
                    overlaps.append((start_time, end_time, overlap_confidence))
        
        # Merge nearby overlaps
        merged_overlaps = self._merge_overlaps(overlaps, merge_distance=0.5)
        return merged_overlaps
    
    def detect_overlap_with_wav2vec(self, audio: np.ndarray) -> List[Tuple[float, float, float]]:
        """Detect overlaps using Wav2Vec2 features."""
        if self.processor is None or self.model is None:
            logger.warning("Wav2Vec2 model not available, falling back to energy-based detection")
            return self.detect_multiple_speakers_energy(audio)
        
        overlaps = []
        window_duration = 2.0  # 2 seconds
        window_samples = int(window_duration * self.sample_rate)
        hop_samples = window_samples // 2
        
        try:
            for i in range(0, len(audio) - window_samples, hop_samples):
                window = audio[i:i + window_samples]
                
                # Process with Wav2Vec2
                inputs = self.processor(window, sampling_rate=self.sample_rate, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state.squeeze().numpy()
                
                # Analyze feature variance and patterns
                feature_var = np.var(features, axis=0)
                high_var_features = np.sum(feature_var > np.mean(feature_var))
                
                # Check for non-uniform patterns indicating overlap
                temporal_patterns = np.diff(features, axis=0)
                pattern_irregularity = np.mean(np.abs(temporal_patterns))
                
                # Calculate overlap confidence
                overlap_confidence = min(1.0, (high_var_features + pattern_irregularity) / 100)
                
                if overlap_confidence > self.overlap_threshold:
                    start_time = i / self.sample_rate
                    end_time = (i + window_samples) / self.sample_rate
                    overlaps.append((start_time, end_time, overlap_confidence))
        
        except Exception as e:
            logger.error(f"Error in Wav2Vec2 overlap detection: {e}")
            return self.detect_multiple_speakers_energy(audio)
        
        return self._merge_overlaps(overlaps, merge_distance=0.5)
    
    def _merge_overlaps(self, overlaps: List[Tuple[float, float, float]], merge_distance: float = 0.5) -> List[Tuple[float, float, float]]:
        """Merge nearby overlap detections."""
        if not overlaps:
            return []
        
        # Sort by start time
        overlaps = sorted(overlaps, key=lambda x: x[0])
        merged = [overlaps[0]]
        
        for start, end, confidence in overlaps[1:]:
            last_start, last_end, last_confidence = merged[-1]
            
            # Merge if close enough
            if start - last_end <= merge_distance:
                merged[-1] = (
                    last_start,
                    max(end, last_end),
                    max(confidence, last_confidence)
                )
            else:
                merged.append((start, end, confidence))
        
        return merged
    
    def analyze_audio_file(self, audio_path: Path) -> Dict[str, Any]:
        """
        Analyze an audio file for overlapping speech.
        
        Returns:
            Dictionary containing overlap analysis results
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration = len(audio) / sr
        
        logger.info(f"Analyzing {audio_path.name} ({duration:.2f}s) for overlaps")
        
        # Detect overlaps using multiple methods
        energy_overlaps = self.detect_multiple_speakers_energy(audio)
        wav2vec_overlaps = self.detect_overlap_with_wav2vec(audio)
        
        # Combine results (take intersection for higher confidence)
        combined_overlaps = []
        
        for e_start, e_end, e_conf in energy_overlaps:
            for w_start, w_end, w_conf in wav2vec_overlaps:
                # Check for overlap between detections
                overlap_start = max(e_start, w_start)
                overlap_end = min(e_end, w_end)
                
                if overlap_start < overlap_end:
                    # There's overlap between the two detection methods
                    combined_conf = (e_conf + w_conf) / 2
                    combined_overlaps.append((overlap_start, overlap_end, combined_conf))
        
        # If no combined overlaps, use energy-based as fallback
        if not combined_overlaps:
            combined_overlaps = energy_overlaps
        
        # Calculate statistics
        total_overlap_duration = sum(end - start for start, end, _ in combined_overlaps)
        overlap_percentage = (total_overlap_duration / duration) * 100 if duration > 0 else 0
        
        return {
            'file': str(audio_path),
            'duration': duration,
            'overlaps': combined_overlaps,
            'total_overlap_duration': total_overlap_duration,
            'overlap_percentage': overlap_percentage,
            'num_overlaps': len(combined_overlaps),
            'has_significant_overlap': overlap_percentage > 5.0  # More than 5% overlap
        }
    
    def filter_overlapping_segments(self, segments: List[Path], output_dir: Path) -> Tuple[List[Path], List[Path]]:
        """
        Filter segments to separate clean vs overlapping audio.
        
        Returns:
            Tuple of (clean_segments, overlapping_segments)
        """
        clean_segments = []
        overlapping_segments = []
        
        output_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = output_dir / "clean"
        overlap_dir = output_dir / "overlapping"
        clean_dir.mkdir(exist_ok=True)
        overlap_dir.mkdir(exist_ok=True)
        
        for segment_path in segments:
            analysis = self.analyze_audio_file(segment_path)
            
            if analysis['has_significant_overlap']:
                # Move to overlapping directory
                overlap_path = overlap_dir / segment_path.name
                shutil.copy2(str(segment_path), str(overlap_path))
                overlapping_segments.append(overlap_path)
                logger.info(f"Overlap detected in {segment_path.name}: {analysis['overlap_percentage']:.1f}%")
            else:
                # Keep in clean directory
                clean_path = clean_dir / segment_path.name
                shutil.copy2(str(segment_path), str(clean_path))
                clean_segments.append(clean_path)
        
        logger.info(f"Filtered segments: {len(clean_segments)} clean, {len(overlapping_segments)} overlapping")
        return clean_segments, overlapping_segments


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    detector = OverlapDetector()
    # analysis = detector.analyze_audio_file(Path("input.flac"))
    # print(f"Overlap analysis: {analysis}")
