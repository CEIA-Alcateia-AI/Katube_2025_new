"""
MOS-Bench/SHEET Audio Quality Filter
Filters audio segments based on speech quality assessment (MOS scores 1-5)
"""
import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

logger = logging.getLogger(__name__)

class MOSQualityFilter:
    """
    Audio quality filter using MOS-Bench/SHEET for speech quality assessment.
    Filters out low-quality audio segments (MOS < 2.0).
    """
    
    def __init__(self, 
                 mos_threshold: float = 2.5,
                 sample_rate: int = 24000,
                 use_cuda: bool = False):
        """
        Initialize MOS quality filter.
        
        Args:
            mos_threshold: Minimum MOS score to accept (default: 2.0)
            sample_rate: Target sample rate for audio processing
            use_cuda: Whether to use GPU acceleration
        """
        self.mos_threshold = mos_threshold
        self.sample_rate = sample_rate
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained MOS predictor from SHEET
        self.predictor = None
        self._load_mos_predictor()
        
    def _load_mos_predictor(self):
        """Load pre-trained MOS predictor from SHEET - OBRIGATÓRIO."""
        try:
            logger.info("🔄 Loading MOS predictor from SHEET (OBRIGATÓRIO)...")
            
            # PyTorch é obrigatório para o filtro MOS
            if not self._check_torch_availability():
                raise RuntimeError("❌ PyTorch é OBRIGATÓRIO para o filtro MOS. Instale: pip install torch torchaudio")
            
            # Load pre-trained model using torch.hub
            self.predictor = torch.hub.load(
                "unilight/sheet:v0.1.0", 
                "default", 
                trust_repo=True, 
                force_reload=False
            )
            
            if self.use_cuda and torch.cuda.is_available():
                self.predictor.model.cuda()
                logger.info("✅ MOS predictor loaded on GPU")
            else:
                logger.info("✅ MOS predictor loaded on CPU")
                
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO: Não foi possível carregar o filtro MOS: {e}")
            logger.error("🔧 Solução: pip install torch torchaudio")
            raise RuntimeError(f"Filtro MOS é OBRIGATÓRIO e falhou ao carregar: {e}")
    
    def _check_torch_availability(self) -> bool:
        """Check if PyTorch is available and working."""
        try:
            import torch
            import torchaudio
            # Test basic functionality
            test_tensor = torch.randn(1000)
            return True
        except ImportError:
            logger.warning("⚠️ PyTorch/torchaudio not installed")
            return False
        except Exception as e:
            logger.warning(f"⚠️ PyTorch error: {e}")
            return False
    
    def predict_mos_score(self, audio_path: Path) -> float:
        """
        Predict MOS score for audio file using SHEET predictor (OBRIGATÓRIO).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MOS score (1.0-5.0)
        """
        try:
            if self.predictor is None:
                raise RuntimeError("❌ Filtro MOS não foi inicializado corretamente")
            
            # SEMPRE usar SHEET predictor (obrigatório)
            return self.predictor.predict(wav_path=str(audio_path))
                
        except Exception as e:
            logger.error(f"❌ ERRO ao predizer MOS para {audio_path.name}: {e}")
            logger.error("🔧 Verifique se PyTorch está instalado: pip install torch torchaudio")
            raise RuntimeError(f"Falha na predição MOS (obrigatória): {e}")
    
    def _simple_quality_assessment(self, audio_path: Path) -> float:
        """
        Simple quality assessment based on audio characteristics.
        Fallback when SHEET predictor is not available.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Estimated MOS score (1.0-5.0)
        """
        try:
            # Try to load audio with different methods
            audio, sr = self._load_audio_flexible(audio_path)
            
            if audio is None:
                logger.warning(f"⚠️ Could not load audio: {audio_path.name}")
                return 1.0
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = self._resample_audio(audio, sr, self.sample_rate)
            
            # Convert to mono
            if len(audio.shape) > 1 and audio.shape[0] > 1:
                audio = np.mean(audio, axis=0)
            
            # Calculate quality metrics
            snr_estimate = self._estimate_snr(audio)
            dynamic_range = self._calculate_dynamic_range(audio)
            spectral_centroid = self._calculate_spectral_centroid(audio)
            silence_ratio = self._calculate_silence_ratio(audio)
            
            # Simple scoring based on metrics
            score = 1.0
            
            # SNR contribution (0-2 points)
            if snr_estimate > 20:
                score += 2.0
            elif snr_estimate > 15:
                score += 1.5
            elif snr_estimate > 10:
                score += 1.0
            elif snr_estimate > 5:
                score += 0.5
            
            # Dynamic range contribution (0-1 point)
            if dynamic_range > 40:
                score += 1.0
            elif dynamic_range > 25:
                score += 0.5
            
            # Spectral centroid contribution (0-1 point)
            if 1000 < spectral_centroid < 3000:  # Good speech range
                score += 1.0
            elif 500 < spectral_centroid < 4000:
                score += 0.5
            
            # Silence ratio penalty (0-1 point)
            if silence_ratio > 0.5:  # Too much silence
                score -= 0.5
            elif silence_ratio > 0.3:
                score -= 0.2
            
            return max(min(score, 5.0), 1.0)  # Clamp between 1.0 and 5.0
            
        except Exception as e:
            logger.warning(f"⚠️ Error in simple quality assessment: {e}")
            return 1.0
    
    def _load_audio_flexible(self, audio_path: Path) -> Tuple[Optional[np.ndarray], int]:
        """Load audio using multiple methods as fallback."""
        try:
            # Try torchaudio first
            if torch is not None:
                waveform, sr = torchaudio.load(str(audio_path))
                return waveform.squeeze().numpy(), sr
        except:
            pass
        
        try:
            # Try librosa as fallback
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=None)
            return audio, sr
        except:
            pass
        
        try:
            # Try soundfile as last resort
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            return audio, sr
        except:
            pass
        
        return None, 0
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            if torch is not None:
                # Use torchaudio resampler
                resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
                audio_tensor = torch.from_numpy(audio).float()
                resampled = resampler(audio_tensor)
                return resampled.numpy()
            else:
                # Use librosa resampling
                import librosa
                return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except Exception as e:
            logger.warning(f"⚠️ Resampling failed: {e}")
            return audio
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        try:
            # Use RMS as signal estimate
            signal_power = np.mean(audio ** 2)
            
            # Estimate noise from quiet parts
            quiet_threshold = np.percentile(np.abs(audio), 10)
            quiet_samples = audio[np.abs(audio) < quiet_threshold]
            
            if len(quiet_samples) > 0:
                noise_power = np.mean(quiet_samples ** 2)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                return max(snr, 0)
            else:
                return 20.0  # Assume good SNR if no quiet parts
                
        except:
            return 10.0  # Default moderate SNR
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        try:
            max_val = np.max(np.abs(audio))
            min_val = np.min(np.abs(audio[np.abs(audio) > 0]))
            dynamic_range = 20 * np.log10(max_val / (min_val + 1e-10))
            return dynamic_range
        except:
            return 20.0  # Default moderate dynamic range
    
    def _calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """Calculate ratio of silence in audio."""
        try:
            # Simple silence detection based on RMS
            frame_length = 1024
            hop_length = frame_length // 4
            
            # Calculate RMS for each frame
            rms_values = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_values.append(rms)
            
            if not rms_values:
                return 0.0
            
            rms_values = np.array(rms_values)
            
            # Define silence threshold (adaptive)
            silence_threshold = np.percentile(rms_values, 20)  # Bottom 20% as silence
            
            # Count silent frames
            silent_frames = np.sum(rms_values < silence_threshold)
            total_frames = len(rms_values)
            
            return silent_frames / total_frames if total_frames > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating silence ratio: {e}")
            return 0.0
    
    def _calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid (brightness)."""
        try:
            # Simple FFT-based spectral centroid
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
            
            # Only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # Calculate centroid
            centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
            return centroid
        except:
            return 2000.0  # Default speech-like centroid
    
    def filter_audio_segments(self, segment_paths: List[Path], rejected_dir: Optional[Path] = None) -> Tuple[List[Path], List[Path]]:
        """
        Filter audio segments based on MOS quality scores.
        
        Args:
            segment_paths: List of audio segment paths
            rejected_dir: Directory to save rejected segments (optional)
            
        Returns:
            Tuple of (accepted_segments, rejected_segments)
        """
        accepted_segments = []
        rejected_segments = []
        rejected_files = []
        
        logger.info(f"🔍 Filtering {len(segment_paths)} audio segments (MOS threshold: {self.mos_threshold})")
        
        for i, segment_path in enumerate(segment_paths):
            try:
                # Predict MOS score
                mos_score = self.predict_mos_score(segment_path)
                
                logger.info(f"📊 {segment_path.name}: MOS = {mos_score:.2f}")
                
                if mos_score >= self.mos_threshold:
                    accepted_segments.append(segment_path)
                    logger.info(f"✅ Accepted: {segment_path.name} (MOS: {mos_score:.2f})")
                else:
                    rejected_segments.append(segment_path)
                    logger.warning(f"❌ Rejected: {segment_path.name} (MOS: {mos_score:.2f} < {self.mos_threshold})")
                    
                    # Save rejected segment to rejected directory
                    if rejected_dir:
                        try:
                            rejected_dir.mkdir(parents=True, exist_ok=True)
                            rejected_filename = f"mos_rejected_{mos_score:.2f}_{segment_path.name}"
                            rejected_path = rejected_dir / rejected_filename
                            
                            # Copy the rejected file
                            import shutil
                            shutil.copy2(segment_path, rejected_path)
                            rejected_files.append(rejected_path)
                            
                            logger.debug(f"📁 Saved MOS rejected segment: {rejected_filename}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not save MOS rejected segment {segment_path.name}: {e}")
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"📈 Processed {i + 1}/{len(segment_paths)} segments...")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {segment_path.name}: {e}")
                rejected_segments.append(segment_path)
                
                # Save error segment to rejected directory
                if rejected_dir:
                    try:
                        rejected_dir.mkdir(parents=True, exist_ok=True)
                        rejected_filename = f"mos_error_{segment_path.name}"
                        rejected_path = rejected_dir / rejected_filename
                        
                        import shutil
                        shutil.copy2(segment_path, rejected_path)
                        rejected_files.append(rejected_path)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not save MOS error segment {segment_path.name}: {e}")
        
        logger.info(f"🎯 Filtering complete: {len(accepted_segments)} accepted, {len(rejected_segments)} rejected")
        if rejected_files:
            logger.info(f"📁 MOS rejected files saved: {len(rejected_files)}")
        
        return accepted_segments, rejected_segments
    
    def get_quality_report(self, segment_paths: List[Path]) -> dict:
        """
        Generate quality report for audio segments.
        
        Args:
            segment_paths: List of audio segment paths
            
        Returns:
            Dictionary with quality statistics
        """
        scores = []
        
        for segment_path in segment_paths:
            try:
                score = self.predict_mos_score(segment_path)
                scores.append(score)
            except:
                scores.append(1.0)
        
        if not scores:
            return {
                'total_segments': 0,
                'average_mos': 0.0,
                'min_mos': 0.0,
                'max_mos': 0.0,
                'accepted_count': 0,
                'rejected_count': 0,
                'acceptance_rate': 0.0
            }
        
        scores = np.array(scores)
        accepted_count = np.sum(scores >= self.mos_threshold)
        
        return {
            'total_segments': len(scores),
            'average_mos': float(np.mean(scores)),
            'min_mos': float(np.min(scores)),
            'max_mos': float(np.max(scores)),
            'accepted_count': int(accepted_count),
            'rejected_count': int(len(scores) - accepted_count),
            'acceptance_rate': float(accepted_count / len(scores))
        }


if __name__ == "__main__":
    # Test the MOS filter
    logging.basicConfig(level=logging.INFO)
    
    filter_instance = MOSQualityFilter()
    
    # Test with sample audio
    test_path = Path("test_audio.flac")
    if test_path.exists():
        mos_score = filter_instance.predict_mos_score(test_path)
        print(f"MOS Score: {mos_score:.2f}")
    else:
        print("No test audio found")
