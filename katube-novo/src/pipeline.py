"""
Main pipeline that orchestrates the complete YouTube audio processing workflow.
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import json
from datetime import datetime

from .config import Config
from .youtube_downloader import YouTubeDownloader
from .audio_segmenter import AudioSegmenter
from .audio_segmenter_optimized import OptimizedAudioSegmenter
from .diarizer import EnhancedDiarizer
from .overlap_detector import OverlapDetector
from .speaker_separator import SpeakerSeparator

logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """
    Complete pipeline for YouTube audio processing:
    1. Download audio from YouTube
    2. Segment audio intelligently
    3. Perform speaker diarization
    4. Detect voice overlaps
    5. Separate audio by speakers
    6. Prepare for STT processing
    """
    
    def __init__(self, 
                 output_base_dir: Optional[Path] = None,
                 huggingface_token: Optional[str] = None,
                 segment_min_duration: float = 10.0,
                 segment_max_duration: float = 15.0):
        
        # Set up directories
        self.output_base_dir = output_base_dir or Config.OUTPUT_DIR
        Config.create_directories()
        
        # Initialize components
        self.downloader = YouTubeDownloader()
        # Use optimized segmenter for better performance
        self.segmenter = OptimizedAudioSegmenter(segment_min_duration, segment_max_duration)
        self.diarizer = EnhancedDiarizer(huggingface_token)
        self.overlap_detector = OverlapDetector()
        self.speaker_separator = SpeakerSeparator()
        
        # Pipeline state
        self.current_session = None
        self.session_dir = None
        
    def create_session(self, session_name: Optional[str] = None) -> Path:
        """Create a new processing session directory."""
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"
        
        self.current_session = session_name
        self.session_dir = self.output_base_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['downloads', 'segments', 'diarization', 'speakers', 'clean', 'overlapping', 'stt_ready']
        for subdir in subdirs:
            (self.session_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Created session: {self.current_session}")
        return self.session_dir
    
    def download_youtube_audio(self, url: str, custom_filename: Optional[str] = None) -> Path:
        """
        Step 1: Download audio from YouTube.
        
        Args:
            url: YouTube URL
            custom_filename: Optional custom filename
            
        Returns:
            Path to downloaded audio file
        """
        logger.info("=== STEP 1: DOWNLOADING YOUTUBE AUDIO ===")
        
        if not self.session_dir:
            raise ValueError("No active session. Call create_session() first.")
        
        # Set download directory to session downloads folder
        self.downloader.output_dir = self.session_dir / 'downloads'
        
        # Download audio
        audio_path = self.downloader.download(url, custom_filename)
        
        logger.info(f"Downloaded: {audio_path}")
        return audio_path
    
    def segment_audio(self, audio_path: Path, use_intelligent_segmentation: bool = True) -> List[Path]:
        """
        Step 2: Segment audio into manageable chunks.
        
        Args:
            audio_path: Path to input audio file
            use_intelligent_segmentation: Use intelligent segmentation vs simple chunking
            
        Returns:
            List of segment file paths
        """
        logger.info("=== STEP 2: SEGMENTING AUDIO ===")
        
        segments_dir = self.session_dir / 'segments'
        
        if use_intelligent_segmentation:
            # Use optimized segmentation method for better performance with large files
            segments = self.segmenter.simple_segment_audio(audio_path, segments_dir)
        else:
            # Simple time-based segmentation fallback
            segments = self._simple_segment_audio(audio_path, segments_dir)
        
        logger.info(f"Created {len(segments)} segments")
        return segments
    
    def perform_diarization(self, segments: List[Path], num_speakers: Optional[int] = None) -> Dict[str, Any]:
        """
        Step 3: Perform speaker diarization on segments.
        
        Args:
            segments: List of audio segment paths
            num_speakers: Hint for number of speakers
            
        Returns:
            Dictionary with diarization results
        """
        logger.info("=== STEP 3: PERFORMING SPEAKER DIARIZATION ===")
        
        diarization_dir = self.session_dir / 'diarization'
        
        # Process segments in batch
        results = self.diarizer.diarize_batch(
            segments, 
            diarization_dir, 
            save_rttm=True
        )
        
        # Summarize results
        successful = [k for k, v in results.items() if 'error' not in v]
        failed = [k for k, v in results.items() if 'error' in v]
        
        logger.info(f"Diarization completed: {len(successful)} successful, {len(failed)} failed")
        
        if failed:
            logger.warning(f"Failed files: {failed}")
        
        return results
    
    def detect_overlaps(self, segments: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Step 4: Detect and separate overlapping vs clean segments.
        
        Args:
            segments: List of segment paths
            
        Returns:
            Tuple of (clean_segments, overlapping_segments)
        """
        logger.info("=== STEP 4: DETECTING VOICE OVERLAPS ===")
        
        overlap_dir = self.session_dir / 'overlapping'
        clean_dir = self.session_dir / 'clean'
        
        # Filter segments based on overlap detection
        clean_segments, overlapping_segments = self.overlap_detector.filter_overlapping_segments(
            segments, self.session_dir
        )
        
        logger.info(f"Overlap detection: {len(clean_segments)} clean, {len(overlapping_segments)} overlapping")
        
        return clean_segments, overlapping_segments
    
    def separate_speakers(self, diarization_results: Dict[str, Any], enhance_audio: bool = True) -> Dict[str, Any]:
        """
        Step 5: Separate audio by speakers based on diarization results.
        
        Args:
            diarization_results: Results from diarization step
            enhance_audio: Apply audio enhancement
            
        Returns:
            Dictionary with speaker separation results
        """
        logger.info("=== STEP 5: SEPARATING SPEAKERS ===")
        
        speakers_dir = self.session_dir / 'speakers'
        separation_results = {}
        
        for audio_path_str, diar_result in diarization_results.items():
            if 'error' in diar_result:
                continue
            
            try:
                audio_path = Path(audio_path_str)
                rttm_path = Path(diar_result['rttm_path']) if diar_result.get('rttm_path') else None
                
                if not rttm_path or not rttm_path.exists():
                    logger.warning(f"No RTTM file for {audio_path.name}")
                    continue
                
                # Process with speaker separator
                result = self.speaker_separator.process_audio_file(
                    audio_path, 
                    rttm_path, 
                    speakers_dir / audio_path.stem,
                    enhance=enhance_audio,
                    create_compilations=True
                )
                
                separation_results[audio_path_str] = result
                
            except Exception as e:
                logger.error(f"Speaker separation failed for {audio_path_str}: {e}")
                separation_results[audio_path_str] = {'error': str(e)}
        
        # Summarize results
        total_speakers = sum(r.get('num_speakers', 0) for r in separation_results.values() if 'error' not in r)
        total_segments = sum(r.get('total_segments', 0) for r in separation_results.values() if 'error' not in r)
        
        logger.info(f"Speaker separation: {total_speakers} speakers, {total_segments} segments")
        
        return separation_results
    
    def prepare_for_stt(self, separation_results: Dict[str, Any]) -> Dict[str, List[Path]]:
        """
        Step 6: Prepare final audio files for STT processing.
        
        Args:
            separation_results: Results from speaker separation
            
        Returns:
            Dictionary of STT-ready files organized by speaker
        """
        logger.info("=== STEP 6: PREPARING FOR STT ===")
        
        stt_dir = self.session_dir / 'stt_ready'
        stt_files = {}
        
        # Collect all speaker files
        for audio_result in separation_results.values():
            if 'error' in audio_result:
                continue
            
            # Use compilation files if available, otherwise use individual segments
            if 'compilation_files' in audio_result:
                for speaker, comp_files in audio_result['compilation_files'].items():
                    if speaker not in stt_files:
                        stt_files[speaker] = []
                    
                    if isinstance(comp_files, list):
                        stt_files[speaker].extend(comp_files)
                    else:
                        stt_files[speaker].append(comp_files)
            
            elif 'speaker_files' in audio_result:
                for speaker, speaker_file_list in audio_result['speaker_files'].items():
                    if speaker not in stt_files:
                        stt_files[speaker] = []
                    stt_files[speaker].extend(speaker_file_list)
        
        # Copy files to STT directory and organize
        organized_files = {}
        for speaker, files in stt_files.items():
            speaker_stt_dir = stt_dir / f"speaker_{speaker}"
            speaker_stt_dir.mkdir(exist_ok=True)
            
            organized_files[speaker] = []
            for file_path in files:
                if isinstance(file_path, Path) and file_path.exists():
                    # Copy to STT directory
                    dest_path = speaker_stt_dir / file_path.name
                    if not dest_path.exists():
                        import shutil
                        shutil.copy2(file_path, dest_path)
                    organized_files[speaker].append(dest_path)
        
        # Log summary
        total_files = sum(len(files) for files in organized_files.values())
        logger.info(f"STT preparation: {len(organized_files)} speakers, {total_files} files ready")
        
        return organized_files
    
    def process_youtube_url(self, 
                           url: str, 
                           custom_filename: Optional[str] = None,
                           num_speakers: Optional[int] = None,
                           enhance_audio: bool = True,
                           use_intelligent_segmentation: bool = True,
                           session_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: process a YouTube URL through all steps.
        
        Args:
            url: YouTube URL
            custom_filename: Custom filename for downloaded audio
            num_speakers: Hint for number of speakers
            enhance_audio: Apply audio enhancement
            use_intelligent_segmentation: Use intelligent vs simple segmentation
            session_name: Custom session name
            
        Returns:
            Dictionary with complete processing results
        """
        start_time = time.time()
        
        logger.info("=== STARTING COMPLETE PIPELINE ===")
        logger.info(f"URL: {url}")
        
        try:
            # Create session
            session_dir = self.create_session(session_name)
            
            # Step 1: Download
            audio_path = self.download_youtube_audio(url, custom_filename)
            
            # Step 2: Segment
            segments = self.segment_audio(audio_path, use_intelligent_segmentation)
            
            # Step 3: Diarization
            diarization_results = self.perform_diarization(segments, num_speakers)
            
            # Step 4: Overlap detection
            clean_segments, overlapping_segments = self.detect_overlaps(segments)
            
            # Step 5: Speaker separation
            separation_results = self.separate_speakers(diarization_results, enhance_audio)
            
            # Step 6: STT preparation
            stt_files = self.prepare_for_stt(separation_results)
            
            # Final results
            processing_time = time.time() - start_time
            
            results = {
                'session_name': self.current_session,
                'session_dir': str(session_dir),
                'url': url,
                'processing_time': processing_time,
                'downloaded_audio': str(audio_path),
                'num_segments': len(segments),
                'num_clean_segments': len(clean_segments),
                'num_overlapping_segments': len(overlapping_segments),
                'diarization_results': diarization_results,
                'separation_results': separation_results,
                'stt_ready_files': stt_files,
                'statistics': self._generate_statistics(stt_files, separation_results)
            }
            
            # Save results to JSON
            results_file = session_dir / 'pipeline_results.json'
            with open(results_file, 'w') as f:
                # Convert Path objects to strings for JSON serialization
                json_results = self._prepare_for_json(results)
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Processing time: {processing_time:.2f}s")
            logger.info(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _simple_segment_audio(self, audio_path: Path, output_dir: Path) -> List[Path]:
        """Simple time-based segmentation fallback."""
        import librosa
        import soundfile as sf
        
        audio, sr = librosa.load(audio_path, sr=self.segmenter.sample_rate, mono=True)
        duration = len(audio) / sr
        
        segments = []
        segment_duration = (self.segmenter.min_duration + self.segmenter.max_duration) / 2
        
        for i, start in enumerate(range(0, int(duration), int(segment_duration))):
            end = min(start + segment_duration, duration)
            
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            segment_audio = audio[start_sample:end_sample]
            
            filename = f"{audio_path.stem}_segment_{i:04d}.{Config.AUDIO_FORMAT}"
            segment_path = output_dir / filename
            
            sf.write(segment_path, segment_audio, sr)
            segments.append(segment_path)
        
        return segments
    
    def _generate_statistics(self, stt_files: Dict[str, List[Path]], 
                           separation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing statistics."""
        import soundfile as sf
        
        stats = {
            'num_speakers': len(stt_files),
            'total_stt_files': sum(len(files) for files in stt_files.values()),
            'speakers': {}
        }
        
        for speaker, files in stt_files.items():
            total_duration = 0
            for file_path in files:
                try:
                    if isinstance(file_path, Path) and file_path.exists():
                        with sf.SoundFile(file_path) as f:
                            total_duration += len(f) / f.samplerate
                except:
                    pass
            
            stats['speakers'][speaker] = {
                'num_files': len(files),
                'total_duration': total_duration,
                'avg_file_duration': total_duration / len(files) if files else 0
            }
        
        return stats
    
    def _prepare_for_json(self, obj):
        """Recursively convert Path objects to strings for JSON serialization."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    pipeline = AudioProcessingPipeline()
    
    # Example: process a YouTube video
    # results = pipeline.process_youtube_url(
    #     "https://www.youtube.com/watch?v=example",
    #     custom_filename="example_video",
    #     num_speakers=2
    # )
    # print(f"Pipeline results: {results['statistics']}")
