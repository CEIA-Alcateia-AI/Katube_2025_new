"""
Enhanced speaker diarization using pyannote with improved processing.
"""
import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment, Timeline
import soundfile as sf
from tqdm import tqdm

from .config import Config

logger = logging.getLogger(__name__)

class EnhancedDiarizer:
    """
    Enhanced speaker diarization using pyannote.
    
    IMPORTANTE: Este diarizer retorna APENAS timestamps de speaker.
    O áudio original NÃO deve ser processado através deste pipeline para
    evitar perda de qualidade devido à reamostragem para 16kHz.
    """
    
    def __init__(self, huggingface_token: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.huggingface_token = huggingface_token or Config.HUGGINGFACE_TOKEN
        self.sample_rate = Config.SAMPLE_RATE
        
        # Initialize pipeline
        self.pipeline = None
        self._load_pipeline()
        
        logger.info(f"Diarizer running on {self.device}")
    
    def _load_pipeline(self):
        """Load the pyannote speaker diarization pipeline."""
        try:
            self.pipeline = Pipeline.from_pretrained(
                Config.PYANNOTE_MODEL,
                use_auth_token=self.huggingface_token
            )
            if self.pipeline is not None:
                self.pipeline = self.pipeline.to(self.device)
                logger.info(f"Loaded {Config.PYANNOTE_MODEL} pipeline")
            else:
                raise ValueError("Pipeline returned None - check model access permissions")
        except Exception as e:
            error_msg = f"Failed to load diarization pipeline: {e}"
            if "gated" in str(e).lower() or "unauthorized" in str(e).lower() or "401" in str(e):
                error_msg += "\n\n🚨 SOLUTION: Visit these URLs and accept terms:\n"
                error_msg += "   • https://hf.co/pyannote/speaker-diarization-3.1\n"
                error_msg += "   • https://hf.co/pyannote/segmentation-3.0\n"
                error_msg += "   • https://hf.co/pyannote/embedding\n"
                error_msg += "Then restart the server."
            logger.error(error_msg)
            self.pipeline = None
            return
    
    def _validate_audio_file(self, audio_path: Path) -> None:
        """
        Validate audio file exists and has supported format.
        
        Args:
            audio_path: Path to audio file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        supported_formats = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus', '.webm'}
        if audio_path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
    
    def diarize_audio(self, audio_path: Path, num_speakers: Optional[int] = None) -> Annotation:
        """
        Perform speaker diarization on an audio file.
        
        IMPORTANTE: pyannote reamostra internamente para 16kHz. Este método
        retorna APENAS timestamps. O áudio original deve ser usado em outras
        etapas do pipeline para manter a qualidade.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (optional hint)
            
        Returns:
            pyannote Annotation object containing only timestamps and speaker labels
            
        Raises:
            RuntimeError: If pipeline is not available
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is unsupported
        """
        if self.pipeline is None:
            error_msg = "Diarization pipeline not available. Please accept model terms and restart server."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Validate input
        self._validate_audio_file(audio_path)
        
        logger.info(f"Diarizing {audio_path.name}")
        
        # Let pyannote load and resample the audio internally
        # DO NOT use the resampled audio in subsequent pipeline stages
        try:
            if num_speakers is not None:
                logger.info(f"Using speaker count hint: {num_speakers}")
                diarization = self.pipeline(str(audio_path), num_speakers=num_speakers)
            else:
                diarization = self.pipeline(str(audio_path))
            
            num_detected = len(diarization.labels())
            logger.info(f"Diarization completed: {num_detected} speakers detected")
            
            if num_detected == 0:
                logger.warning(f"No speakers detected in {audio_path.name}")
            
            return diarization
            
        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")
            raise
    
    def annotation_to_dataframe(self, annotation: Annotation, audio_duration: Optional[float] = None) -> pd.DataFrame:
        """Convert pyannote Annotation to pandas DataFrame."""
        segments_data = []
        
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments_data.append({
                'START': segment.start,
                'END': segment.end,
                'DURATION': segment.duration,
                'SPEAKER': speaker,
                'CONFIDENCE': 1.0  # pyannote doesn't provide confidence in this version
            })
        
        df = pd.DataFrame(segments_data)
        
        if not df.empty:
            # Sort by start time
            df = df.sort_values('START').reset_index(drop=True)
            
            # Add additional metrics
            if audio_duration is not None:
                df['RELATIVE_START'] = df['START'] / audio_duration
                df['RELATIVE_END'] = df['END'] / audio_duration
        
        return df
    
    def save_rttm(self, annotation: Annotation, output_path: Path, audio_filename: str):
        """Save diarization results in RTTM format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            annotation.write_rttm(f)
        logger.info(f"RTTM saved to {output_path}")
    
    def post_process_annotation(
        self, 
        annotation: Annotation, 
        min_duration: float = 0.5,
        collar: float = 0.5
    ) -> Annotation:
        """
        Post-process diarization results.
        
        Args:
            annotation: Original annotation
            min_duration: Minimum segment duration to keep (seconds)
            collar: Maximum gap to merge between segments (seconds)
            
        Returns:
            Processed annotation
        """
        # Remove very short segments
        cleaned = annotation.support(min_duration)
        
        # Merge nearby segments from the same speaker
        processed = Annotation()
        
        for speaker in cleaned.labels():
            speaker_timeline = cleaned.label_timeline(speaker)
            # support() merges segments within collar distance
            merged_timeline = speaker_timeline.support(collar)
            
            for segment in merged_timeline:
                processed[segment] = speaker
        
        logger.info(
            f"Post-processing: {len(list(annotation.itertracks()))} segments -> "
            f"{len(list(processed.itertracks()))} segments"
        )
        
        return processed
    
    def analyze_speaker_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze speaker statistics from diarization results."""
        if df.empty:
            return {'error': 'No diarization data available'}
        
        stats = {}
        
        # Overall statistics
        total_duration = df['DURATION'].sum()
        stats['total_speech_duration'] = total_duration
        stats['num_segments'] = len(df)
        stats['num_speakers'] = df['SPEAKER'].nunique()
        
        # Per-speaker statistics
        speaker_stats = []
        for speaker in df['SPEAKER'].unique():
            speaker_df = df[df['SPEAKER'] == speaker]
            speaker_info = {
                'speaker': speaker,
                'total_duration': speaker_df['DURATION'].sum(),
                'num_segments': len(speaker_df),
                'avg_segment_duration': speaker_df['DURATION'].mean(),
                'speaking_percentage': (speaker_df['DURATION'].sum() / total_duration) * 100
            }
            speaker_stats.append(speaker_info)
        
        # Sort by speaking time
        speaker_stats = sorted(speaker_stats, key=lambda x: x['total_duration'], reverse=True)
        stats['speakers'] = speaker_stats
        
        # Overlap analysis
        overlaps = self._detect_overlaps_in_annotation(df)
        stats['overlaps'] = overlaps
        
        return stats
    
    def _detect_overlaps_in_annotation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect overlaps in the diarization annotation using sweep line algorithm.
        
        This is more efficient than comparing all pairs and catches all overlaps,
        not just consecutive segments.
        """
        if df.empty:
            return {'num_overlaps': 0, 'total_overlap_duration': 0.0}
        
        overlaps = []
        df_sorted = df.sort_values('START').reset_index(drop=True)
        
        # Create events for sweep line algorithm
        events = []
        for idx, row in df_sorted.iterrows():
            events.append(('start', row['START'], row['SPEAKER'], idx, row['END']))
            events.append(('end', row['END'], row['SPEAKER'], idx, row['END']))
        
        # Sort by time, with 'end' events before 'start' events at same time
        events.sort(key=lambda x: (x[1], x[0] == 'start'))
        
        # Track active segments
        active_segments = {}
        
        for event_type, time, speaker, idx, seg_end in events:
            if event_type == 'start':
                # Check for overlaps with all currently active segments
                for active_speaker, active_data in list(active_segments.items()):
                    if active_speaker != speaker:
                        overlap_start = time
                        overlap_end = min(seg_end, active_data['end'])
                        
                        if overlap_end > overlap_start:
                            overlaps.append({
                                'start': overlap_start,
                                'end': overlap_end,
                                'duration': overlap_end - overlap_start,
                                'speakers': sorted([speaker, active_speaker])
                            })
                
                # Add this segment to active segments
                active_segments[speaker] = {
                    'idx': idx,
                    'end': seg_end,
                    'start': time
                }
            else:  # end event
                # Remove from active if this is the matching segment
                if speaker in active_segments and active_segments[speaker]['idx'] == idx:
                    del active_segments[speaker]
        
        total_overlap_duration = sum(o['duration'] for o in overlaps)
        
        return {
            'num_overlaps': len(overlaps),
            'total_overlap_duration': total_overlap_duration,
            'overlap_details': overlaps[:100]  # Limit to first 100 for memory
        }
    
    def diarize_batch(
        self, 
        audio_files: List[Path], 
        output_dir: Path, 
        save_rttm: bool = True,
        show_progress: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Diarize multiple audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save results
            save_rttm: Whether to save RTTM files
            show_progress: Show progress bar
            skip_existing: Skip files with existing RTTM files
            
        Returns:
            Dictionary with batch processing results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        iterator = tqdm(audio_files, desc="Diarizing") if show_progress else audio_files
        
        for audio_path in iterator:
            try:
                # Skip if RTTM already exists
                rttm_path = output_dir / f"{audio_path.stem}.rttm"
                if skip_existing and rttm_path.exists() and save_rttm:
                    logger.info(f"RTTM already exists for {audio_path.name}, skipping")
                    results[str(audio_path)] = {
                        'skipped': True,
                        'rttm_path': str(rttm_path)
                    }
                    continue
                
                # Perform diarization
                annotation = self.diarize_audio(audio_path)
                
                # Convert to DataFrame
                audio_duration = self._get_audio_duration(audio_path)
                df = self.annotation_to_dataframe(annotation, audio_duration)
                
                # Post-process
                processed_annotation = self.post_process_annotation(annotation)
                
                # Save RTTM if requested
                if save_rttm:
                    self.save_rttm(processed_annotation, rttm_path, audio_path.name)
                
                # Analyze statistics
                stats = self.analyze_speaker_statistics(df)
                
                results[str(audio_path)] = {
                    'annotation': processed_annotation,
                    'dataframe': df,
                    'statistics': stats,
                    'rttm_path': str(rttm_path) if save_rttm else None,
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}", exc_info=True)
                results[str(audio_path)] = {
                    'error': str(e),
                    'success': False
                }
        
        # Summary statistics
        successful = sum(1 for r in results.values() if r.get('success', False))
        skipped = sum(1 for r in results.values() if r.get('skipped', False))
        failed = len(results) - successful - skipped
        
        logger.info(
            f"Batch processing complete: {successful} successful, "
            f"{skipped} skipped, {failed} failed"
        )
        
        return results
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            with sf.SoundFile(audio_path) as f:
                return len(f) / f.samplerate
        except Exception as e:
            logger.warning(f"soundfile failed for {audio_path}, using torchaudio: {e}")
            # Fallback using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform.shape[1] / sample_rate
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
        return False


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Using context manager for automatic cleanup
    with EnhancedDiarizer() as diarizer:
        audio_files = list(Path("input_dir").glob("*.flac"))
        results = diarizer.diarize_batch(
            audio_files, 
            Path("output/"),
            show_progress=True,
            skip_existing=True
        )
        
        # Print summary
        for filepath, result in results.items():
            if result.get('success'):
                stats = result['statistics']
                print(f"\n{Path(filepath).name}:")
                print(f"  Speakers: {stats['num_speakers']}")
                print(f"  Duration: {stats['total_speech_duration']:.2f}s")
                for speaker_stat in stats['speakers']:
                    print(f"    {speaker_stat['speaker']}: "
                          f"{speaker_stat['speaking_percentage']:.1f}%")