"""
Main pipeline that orchestrates the complete YouTube audio processing workflow.
"""
import os
import shutil
import time
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import json
from datetime import datetime
from .config import Config
from .youtube_downloader import YouTubeDownloader
from .youtube_scanner import YouTubeChannelScanner
from .audio_segmenter import AudioSegmenter
from .diarizer import EnhancedDiarizer
from .overlap_detector import OverlapDetector
from .speaker_separator import SpeakerSeparator
from .stt_whisper import WhisperSTTTranscriber
from .stt_wav2vec2 import WAV2VEC2STTTranscriber
from .audio_normalizer import AudioNormalizer
from .validation import create_validation_file
from .marcos_validation.validador_transcricao import create_validation_file as marcos_create_validation_file
from .denoiser import Denoiser
from .sox_normalizer import SoxNormalizer
from .mos_filter import MOSQualityFilter

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
    7. Denoise audio segments
    """
    
    def __init__(self, 
                 output_base_dir: Optional[Path] = None,
                 huggingface_token: Optional[str] = None,
                 segment_min_duration: float = 10.0,
                 segment_max_duration: float = 15.0,
                 mos_threshold: float = 2.5,
                 enable_mos_filter: bool = True,
                 use_cuda: bool = False):
        
        # Set up directories
        self.output_base_dir = output_base_dir or Config.OUTPUT_DIR
        Config.create_directories()
        
        # Initialize components
        self.downloader = YouTubeDownloader()
        # Use intelligent segmenter with VAD for quality cuts
        self.segmenter = AudioSegmenter(segment_min_duration, segment_max_duration)
        self.diarizer = EnhancedDiarizer(huggingface_token)
        self.overlap_detector = OverlapDetector()
        self.speaker_separator = SpeakerSeparator()
        
        # Initialize filters
        # Completeness filter moved to separate file (src/audio_completeness_filter.py)
        self.enable_completeness_filter = False  # DISABLED - moved to separate file
        
        logger.info("\n\n\n🔍 Filtros de áudio:")
        logger.info("   - Filtro de completude: DESABILITADO (arquivo separado)")
        
        # Initialize MOS quality filter (OBRIGATÓRIO)
        self.enable_mos_filter = True  # Sempre habilitado
        logger.info("🔍 Inicializando filtro MOS (OBRIGATÓRIO)...")
        
        try:
            self.mos_filter = MOSQualityFilter(
                mos_threshold=mos_threshold,
                use_cuda=use_cuda
            )
            logger.info("✅ Filtro MOS inicializado com sucesso")
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO: Falha ao inicializar filtro MOS: {e}")
            raise RuntimeError(f"Filtro MOS é OBRIGATÓRIO e falhou: {e}")
        
        # Initialize YouTube scanner
        self.youtube_scanner = YouTubeChannelScanner(
            api_key=Config.YOUTUBE_API_KEY or '',
            base_dir=self.output_base_dir / "youtube_scans"
        )
        
        # Initialize STT transcribers (separated models)
        self.enable_stt = True  # Sempre habilitado
        logger.info("\n\n\n🔍 Inicializando STT transcribers separados...")
        
        try:
            # Initialize Whisper STT
            self.whisper_stt = WhisperSTTTranscriber(
                whisper_model_name="freds0/distil-whisper-large-v3-ptbr",  # Modelo especializado em PT-BR
                device="cuda" if use_cuda else "cpu",
                huggingface_token=huggingface_token
            )
            logger.info("✅ Whisper STT transcriber inicializado com sucesso")
            
            # Initialize WAV2VEC2 STT
            self.wav2vec2_stt = WAV2VEC2STTTranscriber(
                wav2vec2_model_name="lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2",  # Modelo especializado em PT-BR
                device="cuda" if use_cuda else "cpu"
            )
            logger.info("✅ WAV2VEC2 STT transcriber inicializado com sucesso")
            
        except Exception as e:
            logger.warning(f"⚠️ STT transcribers falharam: {e}")
            logger.warning("⚠️ Continuando sem STT - pipeline funcionará normalmente")
            self.enable_stt = False
            self.whisper_stt = None
            self.wav2vec2_stt = None
        
        # Initialize audio normalizer
        self.audio_normalizer = AudioNormalizer(
            target_sample_rate=24000,
            target_format="flac",
            target_channels=1  # Mono
        )
        
        # Initialize denoiser
        self.denoiser = Denoiser(model_name="DeepFilterNet3")
        logger.info("\n\n\n✅ Denoiser (DeepFilterNet3) inicializado com sucesso")
        
        # Initialize Sox normalizer for final processing
        self.sox_normalizer = SoxNormalizer(
            target_sample_rate=48000,
            target_format="flac",
            target_channels=1,
            normalize_gain=True
        )
        logger.info("✅ Sox normalizer inicializado com sucesso")
        
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
        
        # Create minimal subdirectories (only for temporary processing)
        subdirs = ['downloads', 'segments', 'diarization', 'speakers', 'clean', 'overlapping', 'stt_ready']
        for subdir in subdirs:
            (self.session_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"📁 Session local criada: {self.current_session}")
        logger.info(f"Created session: {self.current_session}")
        return self.session_dir

    def cleanup(self, stages_to_clean: Optional[List[str]] = None):
        for stage in stages_to_clean:
            logger.info(f'\n\n\n ==== Limpando a pasta {stage} ===')
            diretories_to_delete = self.session_dir / stage
            if diretories_to_delete.exists():
                try:                    
                    logger.info(f"\n\n[FINAL CLEAN-UP] Deletando a pasta: {diretories_to_delete}")
                    shutil.rmtree(diretories_to_delete)
                    logger.info(f"✅ Sucesso: Diretório de downloads deletado: {diretories_to_delete}")
                except Exception as e:
                    logger.error(f"❌ Falha ao deletar o diretório de downloads: {e}")
    
    def download_youtube_audio(self, url: str, custom_filename: Optional[str] = None) -> Path:
        """
        Step 1: Download audio from YouTube.
        
        Args:
            url: YouTube URL
            custom_filename: Optional custom filename
            
        Returns:
            Path to downloaded audio file
        """
        logger.info("\n\n\n=== STEP 1: DOWNLOADING YOUTUBE AUDIO ===")
        
        if not self.session_dir:
            raise ValueError("No active session. Call create_session() first.")
        
        # Set download directory to session downloads folder
        self.downloader.output_dir = self.session_dir / 'downloads'
        
        # Download audio
        audio_path = self.downloader.download(url, custom_filename)
        
        logger.info(f"Downloaded: {audio_path}")
        
        # Step 1.5: Normalize audio (FLAC, 24kHz, Mono)
        logger.info("\n\n\n=== STEP 1.5: NORMALIZING AUDIO ===")
        normalization_result = self.audio_normalizer.normalize_and_replace(audio_path)
        
        if normalization_result['success']:
            logger.info(f"✅ Audio normalized: {normalization_result['format']}, "
                       f"{normalization_result['sample_rate']}Hz, "
                       f"{normalization_result['channels']} channel(s)")
            logger.info(f"   Size: {normalization_result['size'] / (1024*1024):.1f} MB")
        else:
            logger.error(f"❌ Audio normalization failed: {normalization_result['error']}")
            # Continue with original audio if normalization fails
            logger.warning("⚠️ Continuing with original audio format")
        
        return audio_path
    
    def segment_audio(self, audio_path: Path, use_intelligent_segmentation: bool = True) -> List[Path]:
        """
        Step 2: Segment audio into manageable chunks for local processing.
        
        Args:
            audio_path: Path to input audio file
            use_intelligent_segmentation: Use intelligent segmentation vs simple chunking
            
        Returns:
            List of segment file paths
        """
        logger.info("\n\n\n=== STEP 2: SEGMENTING AUDIO ===")
        
        segments_dir = self.session_dir / 'segments'
        
        if use_intelligent_segmentation:
            # Use intelligent segmentation with VAD for quality cuts
            segments = self.segmenter.segment_audio(audio_path, segments_dir)
        else:
            # Simple time-based segmentation fallback
            segments = self._simple_segment_audio(audio_path, segments_dir)
        
        logger.info(f"Created {len(segments)} segments")
        
        # Retornar segmentos brutos - os filtros serão aplicados no pipeline principal
        return segments
    
    def apply_mos_filter(self, segment_paths: List[Path], rejected_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Apply MOS quality filter to audio segments.
        
        Args:
            segment_paths: List of audio segment paths
            rejected_dir: Directory to save rejected segments (optional)
            
        Returns:
            Dictionary with filtering results
        """
        logger.info("\n\n\n=== STEP 4: APPLYING MOS QUALITY FILTER ===")
        
        if not segment_paths:
            logger.warning("⚠️ No segments to filter")
            return {
                'filtered_segments': [],
                'rejected_segments': [],
                'total_segments': 0,
                'accepted_count': 0,
                'rejected_count': 0,
                'quality_rate': 0.0
            }
        
        # Apply MOS filter with 3-tier classification
        approved_segments, intermediate_segments, rejected_segments = self.mos_filter.filter_audio_segments(segment_paths, output_dir=self.session_dir)
        
        # For pipeline continuation, use approved segments (≥3.0)
        accepted_segments = approved_segments
        
        # Generate quality report
        quality_report = self.mos_filter.get_quality_report(segment_paths)
        logger.info(f"\n\n\n📊 MOS Quality Report: {quality_report}")
        
        # Log detailed results
        logger.info(f"\n\n\n🎯 MOS filtering results:")
        logger.info(f"   - Total segments analyzed: {len(segment_paths)}")
        logger.info(f"   - Accepted segments: {len(accepted_segments)}")
        logger.info(f"   - Rejected segments: {len(rejected_segments)}")
        logger.info(f"   - Quality acceptance rate: {len(accepted_segments)/len(segment_paths):.1%}")
        
        return {
            'filtered_segments': accepted_segments,
            'rejected_segments': rejected_segments,
            'total_segments': len(segment_paths),
            'accepted_count': len(accepted_segments),
            'rejected_count': len(rejected_segments),
            'quality_rate': len(accepted_segments)/len(segment_paths) if segment_paths else 0.0,
            'quality_report': quality_report
        }
    
    def filter_segments_by_quality(self, segment_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Filter audio segments based on MOS quality scores (OBRIGATÓRIO).
        
        Args:
            segment_paths: List of audio segment paths
            
        Returns:
            Tuple of (accepted_segments, rejected_segments)
        """
        if self.mos_filter is None:
            raise RuntimeError("❌ Filtro MOS não foi inicializado (OBRIGATÓRIO)")
        
        logger.info(f"\n\n\n🔍 Filtrando {len(segment_paths)} segmentos por qualidade MOS (OBRIGATÓRIO)...")
        
        # Criar pastas específicas para áudios descartados
        rejected_completeness_dir = self.session_dir / 'audio_descartado_completude'
        rejected_mos_dir = self.session_dir / 'audio_descartado_mos'
        
        # Criar as pastas
        rejected_completeness_dir.mkdir(exist_ok=True)
        rejected_mos_dir.mkdir(exist_ok=True)
        
        # Apply MOS filter with 3-tier classification
        approved_segments, intermediate_segments, rejected_segments = self.mos_filter.filter_audio_segments(segment_paths, output_dir=self.session_dir)
        
        # For pipeline continuation, use approved segments (≥3.0)
        accepted_segments = approved_segments
        
        # Generate quality report
        quality_report = self.mos_filter.get_quality_report(segment_paths)
        logger.info(f"📊 Relatório de Qualidade MOS: {quality_report}")
        
        return accepted_segments, rejected_segments
    
    def scan_youtube_channel(self, channel_url: str) -> Dict[str, Any]:
        """
        Scan YouTube channel for all videos.
        
        Args:
            channel_url: YouTube channel URL
            
        Returns:
            Dictionary with scan results
        """
        logger.info(f"🔍 Scanning YouTube channel: {channel_url}")
        
        result = self.youtube_scanner.scan_channel(channel_url)
        
        if result:
            logger.info(f"✅ Channel scan complete: {result}")
            return {
                'success': True,
                'video_list_path': str(result),
                'message': 'Channel scanned successfully'
            }
        else:
            logger.error("❌ Channel scan failed")
            return {
                'success': False,
                'error': 'Channel scan failed',
                'message': 'Could not scan channel'
            }
    
    def process_youtube_channel(self, channel_url: str, max_videos: int = 2500, progress_callback=None) -> Dict[str, Any]:
        """
        Scan and process videos from a YouTube channel.
        
        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum number of videos to process
            progress_callback: Optional callback for progress updates (video_url, success, total_videos, current_index)
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"🔄 Processing YouTube channel: {channel_url}")
        
        def process_video_callback(video_url: str, total_videos: int, current_index: int) -> bool:
            """Callback to process each video from the channel."""
            try:
                logger.info(f"📹 Processing video {current_index}/{total_videos}: {video_url}")
                
                # Process single video through pipeline
                result = self.process_single_video(video_url)
                
                # Update progress if callback provided
                if progress_callback:
                    # Consider it success if we processed it, even if no segments
                    # Local processing success check
                    is_success = result.get('success', False) or result.get('warning') is not None
                    progress_callback(video_url, is_success, total_videos, current_index)
                
                return result.get('success', False)
                
            except Exception as e:
                logger.error(f"❌ Error processing video {video_url}: {e}")
                if progress_callback:
                    progress_callback(video_url, False, total_videos, current_index)
                return False
        
        # Scan and process channel
        result = self.youtube_scanner.scan_and_process_channel(
            channel_url=channel_url,
            process_callback=process_video_callback
        )
        
        return result
    
    def process_single_video(self, video_url: str) -> Dict[str, Any]:
        """
        Process a single YouTube video through the complete pipeline.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"🎬 Processing single video: {video_url}")
            
            # Create session for this video
            self.create_session()
            
            # Step 1: Download video
            try:
                audio_path = self.download_youtube_audio(video_url)
                logger.info(f"✅ Downloaded: {audio_path.name}")
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Download failed: {str(e)}"
                }
            
            # Step 2: Segment audio
            try:
                segments = self.segment_audio(audio_path)
                logger.info(f"✅ Segmented into {len(segments)} segments")
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Segmentation failed: {str(e)}"
                }
            
            # Step 3: Apply completeness filter (DISABLED - moved to separate file)
            # Completeness filter is now in src/audio_completeness_filter.py
            # if self.enable_completeness_filter:
            #     completeness_rejected_dir = self.session_dir / 'audio_descartado_completude'
            #     completeness_result = self.apply_completeness_filter(segments, rejected_dir=completeness_rejected_dir)
            #     segments = completeness_result['complete_segments']
            #     logger.info(f"✅ Completeness filter: {len(segments)} segments passed (filtered {completeness_result['cut_count']} cut segments)")
            
            # Step 4: Apply MOS filter
            if self.enable_mos_filter:
                try:
                    mos_rejected_dir = self.session_dir / 'audio_descartado_mos'
                    mos_result = self.apply_mos_filter(segments, rejected_dir=mos_rejected_dir)
                    segments = mos_result['filtered_segments']
                    logger.info(f"✅ MOS filter: {len(segments)} segments passed")
                except Exception as e:
                    return {
                        'success': False,
                        'error': f"MOS filter failed: {str(e)}"
                    }
            
            # Step 4.5: Move approved segments to final directory
            try:
                final_segments_dir = self.session_dir / 'segments_aprovados'
                final_segments_dir.mkdir(exist_ok=True)
                
                final_segments = []
                for segment in segments:
                    final_path = final_segments_dir / segment.name
                    import shutil
                    shutil.copy2(segment, final_path)
                    final_segments.append(final_path)
                
                segments = final_segments
                logger.info(f"✅ {len(segments)} segments moved to final approved directory")
            except Exception as e:
                logger.warning(f"⚠️ Could not move segments to final directory: {e}")
            
            # Step 5: Perform diarization
            try:
                diarization_result = self.perform_diarization(segments)
                logger.info(f"✅ Diarization completed")
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Diarization failed: {str(e)}"
                }
            
            # Step 6: Detect overlaps
            try:
                overlap_result = self.detect_overlaps(segments)
                logger.info(f"✅ Overlap detection completed")
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Overlap detection failed: {str(e)}"
                }
            
            # Step 7: Separate by speaker
            try:
                separation_result = self.separate_by_speaker(segments, diarization_result['rttm_path'])
                logger.info(f"✅ Speaker separation completed")
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Speaker separation failed: {str(e)}"
                }
            
            # Files are kept locally for processing
            logger.info(f"📁 Files saved locally in: {self.session_dir}")
            
            # Determine success based on whether we have processable content
            success = len(segments) > 0 or len(separation_result.get('stt_files', [])) > 0
            
            return {
                'success': success,
                'video_url': video_url,
                'audio_path': str(audio_path),
                'segments_count': len(segments),
                'speakers_count': diarization_result.get('speakers_count', 0),
                'overlaps_count': overlap_result.get('overlaps_count', 0),
                'stt_files': separation_result.get('stt_files', []),
                'gcp_upload': upload_result,
                'warning': f"Video too short for segmentation ({len(segments)} segments)" if len(segments) == 0 else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing video {video_url}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_video_id(self, video_url: str) -> str:
        """Extract YouTube video ID from URL."""
        import re
        
        # YouTube URL patterns
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)
        
        # Fallback: use hash of URL
        import hashlib
        return hashlib.md5(video_url.encode()).hexdigest()[:11]
    
    def perform_diarization(self, segments: List[Path], num_speakers: Optional[int] = None) -> Dict[str, Any]:
        """
        Step 3: Perform speaker diarization on segments.
        
        Args:
            segments: List of audio segment paths
            num_speakers: Hint for number of speakers
            
        Returns:
            Dictionary with diarization results
        """
        logger.info("\n\n\n=== STEP 3: PERFORMING SPEAKER DIARIZATION ===")
        
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
        logger.info("\n\n\n=== STEP 4: DETECTING VOICE OVERLAPS ===")
        
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
        logger.info("\n\n=== STEP 5: SEPARATING SPEAKERS ===")
        
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

    def denoise_segments(self, segments: List[Path], output_dir: Optional[Path] = None) -> List[Path]:
        """
        Step 7: Denoise audio segments.
        
        Args:
            segments: Lista de arquivos de áudio a serem denoisados
            output_dir: Diretório para salvar os arquivos denoisados
            
        Returns:
            Lista de arquivos denoisados
        """
        logger.info("\n\n\n=== STEP 7: DENOISING AUDIO SEGMENTS AND CLEAN UP ===")
        
        output_dir = output_dir or (self.session_dir / 'clean')
        output_dir.mkdir(exist_ok=True)
        
        denoised_paths = []
        for seg_path in tqdm(segments, desc="Denoising segments"):
            try:
                output_path = output_dir / f"{seg_path.stem}_denoised.{Config.AUDIO_FORMAT}"
                self.denoiser.process_file(seg_path, output_path)
                denoised_paths.append(output_path)
            except Exception as e:
                logger.error(f"Denoising failed for {seg_path.name}: {e}")
        
        logger.info(f"Denoised {len(denoised_paths)}/{len(segments)} segments")

        self.cleanup(stages_to_clean=["downloads", "segments", "stt_ready", "audios_abaixo_2,5_MOS", "audios_acima_3,0_MOS", "audios_validados_tts", "audios_denoiser", "clean", "audios_entre_2,5_e_3,0_MOS", "diarization", "overlapping", "speakers"])

        return denoised_paths
    
    def prepare_for_stt(self, separation_results: Dict[str, Any]) -> Dict[str, List[Path]]:
        """
        Step 6: Prepare final audio files for STT processing.
        
        Args:
            separation_results: Results from speaker separation
            
        Returns:
            Dictionary of STT-ready files organized by speaker
        """
        logger.info("\n\n\n=== STEP 6: PREPARING FOR STT ===")
        
        stt_dir = self.session_dir / 'stt_ready'
        stt_files = {}
        
        # Collect all speaker files
        for audio_result in separation_results.values():
            if 'error' in audio_result:
                continue
            
            # Use individual segments for STT (better for validation), not compilations
            if 'speaker_files' in audio_result:
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
    
    def transcribe_audio_segments(self, 
                                 segment_paths: List[Path]) -> Dict[str, Any]:
        """
        Step 7: Transcribe audio segments using Whisper and WAV2VEC2 (separated models).
        
        Args:
            segment_paths: List of audio segment paths to transcribe
            
        Returns:
            Dictionary with transcription results
        """
        logger.info("\n\n\n=== STEP 7: TRANSCRIBING AUDIO SEGMENTS ===")
        
        if not self.enable_stt:
            logger.warning("STT transcribers are disabled, skipping transcription")
            return {"error": "STT transcribers are disabled"}
        
        try:
            # Create STT output directory
            stt_output_dir = self.session_dir / 'stt_results'
            
            # Transcribe with Whisper
            whisper_results = {}
            if self.whisper_stt:
                logger.info("🎤 Transcribing with Whisper...")
                whisper_results = self.whisper_stt.transcribe_segments(
                    segment_paths=segment_paths,
                    output_dir=stt_output_dir
                )
                logger.info(f"✅ Whisper transcription completed: {whisper_results['whisper_count']} segments")
            
            # Transcribe with WAV2VEC2
            wav2vec2_results = {}
            if self.wav2vec2_stt:
                logger.info("🎤 Transcribing with WAV2VEC2...")
                wav2vec2_results = self.wav2vec2_stt.transcribe_segments(
                    segment_paths=segment_paths,
                    output_dir=stt_output_dir
                )
                logger.info(f"✅ WAV2VEC2 transcription completed: {wav2vec2_results['wav2vec2_count']} segments")
            
            # Combine results
            combined_results = {
                "whisper_results": whisper_results.get("whisper_results", []),
                "wav2vec2_results": wav2vec2_results.get("wav2vec2_results", []),
                "whisper_dir": whisper_results.get("whisper_dir", ""),
                "wav2vec2_dir": wav2vec2_results.get("wav2vec2_dir", ""),
                "total_segments": len(segment_paths),
                "whisper_count": whisper_results.get("whisper_count", 0),
                "wav2vec2_count": wav2vec2_results.get("wav2vec2_count", 0)
            }
            
            logger.info(f"Transcription completed: {combined_results['whisper_count']} Whisper, {combined_results['wav2vec2_count']} WAV2VEC2")
            
            # Step 8: Validate STT transcriptions
            if whisper_results and wav2vec2_results:
                validation_result = self.validate_stt_transcriptions(
                    whisper_results=whisper_results.get("whisper_results", []),
                    wav2vec2_results=wav2vec2_results.get("wav2vec2_results", []),
                    output_dir=stt_output_dir
                )
                combined_results['validation'] = validation_result
                logger.info(f"✅ STT validation completed: {validation_result.get('average_similarity', 0):.3f} avg similarity")
                
                # Step 9: Filter by similarity threshold and apply denoising
                if validation_result.get('success') and validation_result.get('validation_results'):
                    filter_result = self.filter_and_denoise_segments(
                        validation_results=validation_result['validation_results'],
                        output_dir=self.session_dir,  # Use session_dir instead of stt_output_dir
                        similarity_threshold=0.80
                    )
                    combined_results['filter_and_denoise'] = filter_result
                    logger.info(f"✅ Filtering and denoising completed: {filter_result.get('validated_count', 0)} validated, {filter_result.get('denoised_count', 0)} denoised")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in transcription step: {e}")
            return {"error": str(e)}
    
    
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
        
        logger.info("\n\n== STARTING COMPLETE PIPELINE ===")
        logger.info(f"URL: {url}")
        
        try:
            # Create session
            session_dir = self.create_session(session_name)
            
            # Step 1: Download
            audio_path = self.download_youtube_audio(url, custom_filename)
            
            # Step 2: Segment
            segments = self.segment_audio(audio_path, use_intelligent_segmentation)
            
            # Step 3: Apply completeness filter (DISABLED - moved to separate file)
            # Completeness filter is now in src/audio_completeness_filter.py
            # if self.enable_completeness_filter:
            #     completeness_rejected_dir = session_dir / 'audio_descartado_completude'
            #     completeness_result = self.apply_completeness_filter(segments, rejected_dir=completeness_rejected_dir)
            #     segments = completeness_result['complete_segments']
            #     logger.info(f"✅ Completeness filter: {len(segments)} segments passed (filtered {completeness_result['cut_count']} cut segments)")
            
            # Step 4: Apply MOS filter
            if self.enable_mos_filter:
                try:
                    mos_rejected_dir = session_dir / 'audio_descartado_mos'
                    mos_result = self.apply_mos_filter(segments, rejected_dir=mos_rejected_dir)
                    segments = mos_result['filtered_segments']
                    logger.info(f"✅ MOS filter: {len(segments)} segments passed")
                except Exception as e:
                    logger.error(f"❌ MOS filter failed: {e}")
                    return {'success': False, 'error': f"MOS filter failed: {str(e)}"}
            
            # Step 5: Diarization (ANTES do STT)
            diarization_results = self.perform_diarization(segments, num_speakers)
            
            # Step 6: Overlap detection (ANTES do STT)
            clean_segments, overlapping_segments = self.detect_overlaps(segments)
            
            # Step 7: Speaker separation (ANTES do STT)
            separation_results = self.separate_speakers(diarization_results, enhance_audio)
            
            # Step 8: STT preparation (ANTES do STT)
            stt_files = self.prepare_for_stt(separation_results)
            
            # Step 9: Apply STT transcription
            stt_result = {}
            if self.enable_stt:
                try:
                    stt_result = self.transcribe_audio_segments(stt_files if stt_files else segments)
                    logger.info(f"✅ STT transcription completed: {stt_result.get('whisper_count', 0)} Whisper, {stt_result.get('wav2vec2_count', 0)} WAV2VEC2")
                    
                    # Check if validation and filtering were applied
                    if 'validation' in stt_result and 'filter_and_denoise' in stt_result:
                        validation_info = stt_result['validation']
                        filter_info = stt_result['filter_and_denoise']
                        logger.info(f"📊 STT Validation: {validation_info.get('average_similarity', 0):.3f} avg similarity")
                        logger.info(f"📊 Filtro 80%: {filter_info.get('validated_count', 0)} validados, {filter_info.get('denoised_count', 0)} denoised")
                    
                except Exception as e:
                    logger.error(f"❌ STT transcription failed: {e}")
                    # Continue without STT if it fails
                    logger.warning("Continuing pipeline without STT transcription")
            
            # Step 10: Move approved segments to final directory
            try:
                final_segments_dir = session_dir / 'segments_aprovados'
                final_segments_dir.mkdir(exist_ok=True)
                
                final_segments = []
                segments_to_move = stt_files if stt_files else segments
                for segment in segments_to_move:
                    final_path = final_segments_dir / segment.name
                    import shutil
                    shutil.copy2(segment, final_path)
                    final_segments.append(final_path)
                
                segments = final_segments
                logger.info(f"✅ {len(segments)} segments moved to final approved directory")
            except Exception as e:
                logger.warning(f"⚠️ Could not move segments to final directory: {e}")
            
            # Final results
            processing_time = time.time() - start_time

            print("DEBUG: Ponto de verificação #2 - Montando o dicionário de resultados.")
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
                'denoised_segments': [str(p) for p in denoised_segments],
                'stt_ready_files': stt_files,
                'stt_results': stt_result,  # Include STT validation and filtering results
                'statistics': self._generate_statistics(stt_files, separation_results)
            }
            
            print("DEBUG: Ponto de verificação #3 - Preparando para salvar o arquivo JSON.")
            # Save results to JSON
            results_file = session_dir / 'pipeline_results.json'
            with open(results_file, 'w') as f:
                print("DEBUG: Ponto de verificação #4 - Convertendo resultados para o formato JSON.")
                json_results = self._prepare_for_json(results)
                print("DEBUG: Ponto de verificação #5 - Escrevendo o arquivo JSON no disco.")
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            logger.info("\n\n\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
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
    
    # Completeness filter method moved to separate file (src/audio_completeness_filter.py)
    
    def validate_stt_transcriptions(self, 
                                   whisper_results: List[Dict], 
                                   wav2vec2_results: List[Dict],
                                   output_dir: Path) -> Dict[str, Any]:
        """
        Step 8: Validate STT transcriptions using Levenshtein distance.
        
        Args:
            whisper_results: List of Whisper transcription results
            wav2vec2_results: List of WAV2VEC2 transcription results
            output_dir: Directory to save validation results
            
        Returns:
            Dictionary with validation results
        """
        logger.info("\n\n\n=== STEP 8: VALIDATING STT TRANSCRIPTIONS ===")
        
        if not whisper_results or not wav2vec2_results:
            logger.warning("⚠️ No STT results to validate")
            return {"error": "No STT results to validate"}
        
        try:
            # Create validation output directory
            validation_dir = output_dir / 'validation_results'
            validation_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata files for validation (exactly as validation.py expects)
            whisper_metadata_file = validation_dir / 'metadata_whisper.csv'
            wav2vec2_metadata_file = validation_dir / 'metadata_wav2vec2.csv'
            validation_output_file = validation_dir / 'validation_results.csv'
            
            # Validate that both STT models processed the same segments
            if len(whisper_results) != len(wav2vec2_results):
                logger.warning(f"⚠️ Different number of segments: Whisper={len(whisper_results)}, WAV2VEC2={len(wav2vec2_results)}")
                logger.warning("⚠️ Skipping validation - both models must process same segments")
                return {"error": "Different number of segments processed by STT models"}
            
            # Write Whisper metadata (format: filename | text - exactly as validation.py expects)
            with open(whisper_metadata_file, 'w', encoding='utf-8') as f:
                for result in whisper_results:
                    filename = Path(result['file']).stem.replace('_whisper', '').strip()
                    text = result['transcription'].strip()
                    logger.debug(f"Whisper metadata: '{filename}' | '{text[:50]}...'")
                    f.write(f"{filename}|{text}\n")
            
            # Write WAV2VEC2 metadata (format: filename | text - exactly as validation.py expects)  
            with open(wav2vec2_metadata_file, 'w', encoding='utf-8') as f:
                for result in wav2vec2_results:
                    filename = Path(result['file']).stem.replace('_wav2vec2', '').strip()
                    text = result['transcription'].strip()
                    logger.debug(f"WAV2VEC2 metadata: '{filename}' | '{text[:50]}...'")
                    f.write(f"{filename}|{text}\n")
                    
            logger.info(f"📝 Created metadata files:")
            logger.info(f"   - Whisper: {len(whisper_results)} entries")
            logger.info(f"   - WAV2VEC2: {len(wav2vec2_results)} entries")
            
            # Run validation using the professor's validator (exactly as validation.py expects)
            logger.info("🔍 Running STT validation with Levenshtein distance...")
            logger.info(f"   - Input file 1: {whisper_metadata_file}")
            logger.info(f"   - Input file 2: {wav2vec2_metadata_file}")
            logger.info(f"   - Output file: {validation_output_file}")
            
            # Use Marcos validation instead of the old one
            validation_success = marcos_create_validation_file(
                input_file1=str(whisper_metadata_file),
                input_file2=str(wav2vec2_metadata_file),
                prefix_filepath="",  # Empty prefix as in validation.py
                output_file=str(validation_output_file)
            )
            
            # Initialize variables
            validation_results = []
            avg_similarity = 0
            min_similarity = 0
            max_similarity = 0
            
            if validation_success:
                logger.info(f"✅ STT validation completed: {validation_output_file}")
                
                # Read validation results (exactly as validation.py produces)
                validation_results = []
                with open(validation_output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    logger.info(f"📄 Validation file has {len(lines)} lines")
                    
                    if len(lines) > 0:
                        logger.info(f"📄 Header line: '{lines[0].strip()}'")
                        
                    # Skip header if present: filename|subtitle|transcript|similarity
                    data_lines = lines[1:] if len(lines) > 1 else lines
                    
                    for i, line in enumerate(data_lines):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        parts = line.split('|')  # Split by '|' as validation.py uses (no spaces)
                        logger.debug(f"Line {i+1}: '{line}' -> {len(parts)} parts")
                        
                        if len(parts) >= 4:
                            try:
                                similarity = float(parts[3].strip())
                                validation_results.append({
                                    'filename': parts[0].strip(),
                                    'whisper_text': parts[1].strip(),
                                    'wav2vec2_text': parts[2].strip(),
                                    'similarity': similarity
                                })
                                logger.debug(f"✅ Added validation result: {parts[0].strip()} -> {similarity}")
                            except ValueError as e:
                                logger.warning(f"⚠️ Could not parse similarity '{parts[3]}': {e}")
                        else:
                            logger.warning(f"⚠️ Invalid line format (expected 4 parts, got {len(parts)}): '{line}'")
                
                # Calculate statistics
                similarities = [r['similarity'] for r in validation_results]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                min_similarity = min(similarities) if similarities else 0
                max_similarity = max(similarities) if similarities else 0
                
                logger.info(f"📊 Validation statistics:")
                logger.info(f"   - Total segments validated: {len(validation_results)}")
                logger.info(f"   - Average similarity: {avg_similarity:.3f}")
                logger.info(f"   - Min similarity: {min_similarity:.3f}")
                logger.info(f"   - Max similarity: {max_similarity:.3f}")
                
            return {
                    'success': True,
                    'validation_file': str(validation_output_file),
                    'total_segments': len(validation_results),
                    'average_similarity': avg_similarity,
                    'min_similarity': min_similarity,
                    'max_similarity': max_similarity,
                    'validation_results': validation_results
                }
            
            # If validation failed
            if not validation_success:
                logger.error("❌ STT validation failed")
                return {"error": "STT validation failed"}
                
        except Exception as e:
            logger.error(f"Error in STT validation: {e}")
            return {"error": str(e)}
    
    def filter_and_denoise_segments(self, 
                                   validation_results: List[Dict], 
                                   output_dir: Path,
                                   similarity_threshold: float = 0.80) -> Dict[str, Any]:
        """
        Step 9: Filter segments by similarity threshold and apply denoising.
        
        Args:
            validation_results: List of validation results with similarity scores
            output_dir: Directory to save processed segments
            similarity_threshold: Minimum similarity score to accept (default 0.80)
            
        Returns:
            Dictionary with filtering and denoising results
        """
        logger.info(f"\n\n\n=== STEP 9: FILTERING AND DENOISING SEGMENTS (threshold: {similarity_threshold}) ===")
        
        # Temporarily set debug level to see what's happening
        import logging
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        
        if not validation_results:
            logger.warning("⚠️ No validation results to process")
            return {"error": "No validation results to process"}
        
        try:
            # Create output directories following the desired structure
            validated_dir = output_dir / 'audios_validados_tts'
            denoised_dir = output_dir / 'audios_denoiser'
            rejected_dir = output_dir / 'audio_rejeitado_validacao'
            
            validated_dir.mkdir(parents=True, exist_ok=True)
            denoised_dir.mkdir(parents=True, exist_ok=True)
            rejected_dir.mkdir(parents=True, exist_ok=True)
            
            # Filter segments by similarity threshold
            validated_segments = []
            rejected_segments = []
            
            for result in validation_results:
                filename = result['filename']
                similarity = result['similarity']
                
                # Extract base name from validation filename to match with actual audio files
                # Example: segment_000_stt_001 -> segment_000
                base_name = self._extract_base_name_for_validation(filename)
                
                # Search for the actual audio file in stt_ready subdirectories
                audio_file = None
                possible_locations = [
                    # Search in stt_ready subdirectories (speaker_SPEAKER_00, speaker_SPEAKER_01, etc.)
                    output_dir / 'stt_ready' / 'speaker_SPEAKER_00',
                    output_dir / 'stt_ready' / 'speaker_SPEAKER_01',
                    output_dir / 'stt_ready' / 'speaker_SPEAKER_02',
                    output_dir / 'stt_ready' / 'speaker_SPEAKER_03',
                    # Search in actual speaker directories created by prepare_for_stt
                    output_dir / 'stt_ready' / 'speaker_00',
                    output_dir / 'stt_ready' / 'speaker_01',
                    output_dir / 'stt_ready' / 'speaker_02',
                    output_dir / 'stt_ready' / 'speaker_03',
                    # Fallback locations
                    output_dir / 'stt_ready',
                    output_dir / 'speakers',
                    output_dir / 'segments_aprovados',
                    output_dir / 'segments'
                ]
                
                # Search in each possible location
                for location in possible_locations:
                    if location.exists():
                        logger.info(f"🔍 Searching in: {location}")
                        files_found = list(location.glob("*.flac"))
                        logger.info(f"📁 Found {len(files_found)} FLAC files in {location}")
                        
                        # Log first few files for debugging
                        for i, audio_path in enumerate(files_found[:3]):
                            logger.info(f"   File {i+1}: {audio_path.name}")
                        
                        # Search for files in this specific location
                        for audio_path in files_found:
                            logger.debug(f"  Comparing base '{base_name}' with file stem '{audio_path.stem}'")
                            # Check if the base name matches
                            if base_name in audio_path.stem:
                                audio_file = audio_path
                                logger.info(f"✅ Found audio file: {audio_file} (base: {base_name})")
                                break
                        if audio_file:
                            break
                    else:
                        logger.debug(f"❌ Location does not exist: {location}")
                
                if not audio_file:
                    logger.warning(f"⚠️ Could not find audio file for: {filename}")
                    continue
                
                # Categorize based on similarity threshold
                if similarity >= similarity_threshold:
                    validated_segments.append({
                        'filename': filename,
                        'similarity': similarity,
                        'original_path': audio_file,
                        'validated_path': validated_dir / f"{filename}.flac",
                        'denoised_path': denoised_dir / f"{filename}_denoised.flac"
                    })
                    logger.info(f"✅ Validated: {filename} (similarity: {similarity:.3f})")
                else:
                    rejected_segments.append({
                        'filename': filename,
                        'similarity': similarity,
                        'original_path': audio_file,
                        'rejected_path': rejected_dir / f"{filename}.flac"
                    })
                    logger.info(f"❌ Rejected: {filename} (similarity: {similarity:.3f})")
            
            logger.info(f"📊 Filtering results:")
            logger.info(f"   - Total segments: {len(validation_results)}")
            logger.info(f"   - Validated segments (≥{similarity_threshold}): {len(validated_segments)}")
            logger.info(f"   - Rejected segments (<{similarity_threshold}): {len(rejected_segments)}")
            
            # Log details of validated segments
            if validated_segments:
                logger.info("📋 Validated segments details:")
                for seg in validated_segments[:5]:  # Show first 5
                    logger.info(f"   - {seg['filename']}: {seg['similarity']:.3f}")
                if len(validated_segments) > 5:
                    logger.info(f"   ... and {len(validated_segments) - 5} more")
            
            # Log details of rejected segments  
            if rejected_segments:
                logger.info("📋 Rejected segments details:")
                for seg in rejected_segments[:5]:  # Show first 5
                    logger.info(f"   - {seg['filename']}: {seg['similarity']:.3f}")
                if len(rejected_segments) > 5:
                    logger.info(f"   ... and {len(rejected_segments) - 5} more")
            
            # Copy validated segments to audios_validados_tts directory
            logger.info(f"📁 Copying {len(validated_segments)} validated segments to audios_validados_tts...")
            for seg in validated_segments:
                import shutil
                shutil.copy2(seg['original_path'], seg['validated_path'])
                logger.debug(f"✅ Copied to validated: {seg['filename']}")
            
            # Copy rejected segments to audio_rejeitado_validacao directory
            logger.info(f"📁 Copying {len(rejected_segments)} rejected segments to audio_rejeitado_validacao...")
            for seg in rejected_segments:
                import shutil
                shutil.copy2(seg['original_path'], seg['rejected_path'])
                logger.debug(f"❌ Copied to rejected: {seg['filename']}")
            
            # Apply denoising to validated segments and save to audios_denoiser
            logger.info(f"🔊 Applying DeepFilterNet3 denoising to {len(validated_segments)} validated segments...")
            denoised_count = 0
            
            for seg in validated_segments:
                try:
                    logger.info(f"🎛️ Denoising: {seg['filename']} (similarity: {seg['similarity']:.3f})")
                    self.denoiser.process_file(
                        str(seg['validated_path']), 
                        str(seg['denoised_path'])
                    )
                    denoised_count += 1
                    logger.info(f"✅ Denoised and saved to audios_denoiser: {seg['filename']}")
                except Exception as e:
                    logger.error(f"❌ Error denoising {seg['filename']}: {e}")
            
            logger.info(f"✅ Denoising completed: {denoised_count}/{len(validated_segments)} segments processed")
            
            # Restore original log level
            logger.setLevel(original_level)
            
            # Collect denoised audio paths for final dataset creation
            denoised_audio_paths = []
            logger.info(f"🔍 Coletando caminhos de áudios denoised de {len(validated_segments)} segmentos validados...")
            for seg in validated_segments:
                denoised_path = seg['denoised_path']
                logger.debug(f"   Verificando: {denoised_path}")
                if denoised_path.exists():
                    denoised_audio_paths.append(denoised_path)
                    logger.info(f"   ✅ Encontrado: {denoised_path.name}")
                else:
                    logger.warning(f"   ❌ Não encontrado: {denoised_path}")
            
            logger.info(f"📊 Total de áudios denoised coletados: {len(denoised_audio_paths)}")
            
            return {
                'success': True,
                'total_segments': len(validation_results),
                'validated_count': len(validated_segments),
                'rejected_count': len(rejected_segments),
                'denoised_count': denoised_count,
                'similarity_threshold': similarity_threshold,
                'validated_dir': str(validated_dir),
                'denoised_dir': str(denoised_dir),
                'rejected_dir': str(rejected_dir),
                'validated_segments': validated_segments,
                'rejected_segments': rejected_segments,
                'denoised_audio_paths': denoised_audio_paths
            }
            
        except Exception as e:
            logger.error(f"Error in filtering and denoising: {e}")
            return {"error": str(e)}
    
    def _extract_base_name_for_validation(self, filename: str) -> str:
        """
        Extract base name from validation filename for matching with actual audio files.
        
        Examples:
        - segment_000_stt_001 -> segment_000
        - chunk_00_stt_007 -> chunk_00
        
        Args:
            filename: Validation filename with _stt_XXX suffix
            
        Returns:
            Base name without _stt_XXX suffix
        """
        # Remove _stt_XXX pattern from the end
        import re
        base_name = re.sub(r'_stt_\d+$', '', filename)
        logger.debug(f"🔍 Extracted base name: '{filename}' -> '{base_name}'")
        return base_name
    
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
    
    def create_final_dataset(self, denoised_audio_paths: List[Path], stt_results_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Cria o dataset final com normalização Sox e transcrições organizadas.
        
        Args:
            denoised_audio_paths: Lista de caminhos dos áudios denoised
            stt_results_dir: Diretório com resultados STT
            output_dir: Diretório de saída para o dataset final
            
        Returns:
            Dicionário com resultados da criação do dataset
        """
        logger.info("🎯 Criando dataset final com normalização Sox...")
        
        # Criar diretórios para o dataset final
        final_audio_dir = output_dir / "audios_final"
        final_transcriptions_dir = output_dir / "transcricoes_final"
        final_audio_dir.mkdir(parents=True, exist_ok=True)
        final_transcriptions_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'successful_normalizations': [],
            'failed_normalizations': [],
            'transcription_pairs': [],
            'total_processed': len(denoised_audio_paths),
            'success_count': 0,
            'failure_count': 0
        }
        
        logger.info(f"📁 Diretórios criados:")
        logger.info(f"   - Áudios finais: {final_audio_dir}")
        logger.info(f"   - Transcrições finais: {final_transcriptions_dir}")
        
        # Processar cada áudio denoised
        for i, denoised_path in enumerate(denoised_audio_paths):
            try:
                logger.info(f"🔄 Processando {i+1}/{len(denoised_audio_paths)}: {denoised_path.name}")
                
                # Extrair nome base para nomenclatura final
                from .naming_utils import extract_base_name, generate_standard_name
                base_name = extract_base_name(denoised_path)
                
                # Remove "_denoised" suffix para nomenclatura limpa
                if base_name.endswith("_denoised"):
                    base_name = base_name[:-9]  # Remove "_denoised"
                
                # SEMPRE EXECUTAR SOX - Normalizar áudio (24kHz → 48kHz)
                final_audio_name = generate_standard_name(base_name, "final", i+1)
                final_audio_path = final_audio_dir / f"{final_audio_name}.flac"
                
                logger.info(f"🎵 EXECUTANDO SOX: {denoised_path.name} → {final_audio_path.name}")
                print(f"🎵 SOX NORMALIZATION: {denoised_path} → {final_audio_path}")
                
                normalization_result = self.sox_normalizer.normalize_audio(
                    input_path=denoised_path,
                    output_path=final_audio_path
                )
                
                if normalization_result['success']:
                    results['successful_normalizations'].append(normalization_result)
                    results['success_count'] += 1
                    logger.info(f"✅ SOX CONCLUÍDO: {final_audio_path.name}")
                    print(f"✅ SOX SUCCESS: {final_audio_path}")
                    
                    # BUSCAR E COPIAR TRANSCRIÇÕES STT (Whisper + WAV2VEC2)
                    logger.info(f"📝 Buscando transcrições STT para: {base_name}")
                    transcription_files = self._find_transcription_files(base_name, stt_results_dir)
                    
                    if transcription_files:
                        # Copiar transcrições para pasta final
                        final_transcriptions = self._copy_transcriptions_to_final(
                            transcription_files, 
                            final_transcriptions_dir, 
                            final_audio_name
                        )
                        
                        results['transcription_pairs'].append({
                            'audio_file': str(final_audio_path),
                            'transcriptions': final_transcriptions,
                            'base_name': base_name
                        })
                        
                        logger.info(f"✅ {final_audio_path.name} + {len(final_transcriptions)} transcrições copiadas")
                        print(f"📝 TRANSCRIPTIONS COPIED: {len(final_transcriptions)} files for {final_audio_path.name}")
                    else:
                        logger.warning(f"⚠️ Nenhuma transcrição encontrada para {base_name}")
                        print(f"⚠️ NO TRANSCRIPTIONS FOUND for {base_name}")
                        
                else:
                    results['failed_normalizations'].append({
                        'input_path': str(denoised_path),
                        'error': normalization_result['error']
                    })
                    results['failure_count'] += 1
                    logger.error(f"❌ SOX FALHOU: {normalization_result['error']}")
                    print(f"❌ SOX FAILED: {normalization_result['error']}")
                    
            except Exception as e:
                error_msg = f"Erro no processamento de {denoised_path.name}: {str(e)}"
                results['failed_normalizations'].append({
                    'input_path': str(denoised_path),
                    'error': error_msg
                })
                results['failure_count'] += 1
                logger.error(f"❌ {error_msg}")
        
        # Estatísticas finais
        logger.info(f"🎯 Dataset final criado:")
        logger.info(f"   ✅ Sucessos: {results['success_count']}")
        logger.info(f"   ❌ Falhas: {results['failure_count']}")
        logger.info(f"   📝 Pares áudio-transcrição: {len(results['transcription_pairs'])}")
        logger.info(f"   📁 Localização: {output_dir}")
        
        return results
    
    def _find_transcription_files(self, base_name: str, stt_results_dir: Path) -> List[Path]:
        """
        Busca arquivos de transcrição correspondentes a um áudio.
        
        Args:
            base_name: Nome base do arquivo de áudio (ex: segment_000_stt_001)
            stt_results_dir: Diretório com resultados STT
            
        Returns:
            Lista de caminhos dos arquivos de transcrição encontrados
        """
        transcription_files = []
        
        # Buscar em subdiretórios de STT (whisper e wav2vec2) - caminhos corretos
        stt_directories = [
            stt_results_dir / "STT-whisper",
            stt_results_dir / "STT-wav2vec2"
        ]
        
        logger.info(f"🔍 Buscando transcrições para base_name: {base_name}")
        
        for stt_dir in stt_directories:
            logger.info(f"📁 Verificando diretório: {stt_dir}")
            
            if stt_dir.exists():
                # Listar todos os arquivos .txt no diretório
                txt_files = list(stt_dir.glob("*.txt"))
                logger.info(f"   Encontrados {len(txt_files)} arquivos .txt")
                
                for txt_file in txt_files:
                    logger.debug(f"   Verificando arquivo: {txt_file.name}")
                    
                    # Verificar se o nome base está no nome do arquivo
                    if base_name in txt_file.stem or txt_file.stem.startswith(base_name):
                        transcription_files.append(txt_file)
                        logger.info(f"   ✅ MATCH: {txt_file.name}")
                    else:
                        logger.debug(f"   ❌ No match: {txt_file.stem} != {base_name}")
            else:
                logger.warning(f"   ❌ Diretório não existe: {stt_dir}")
        
        logger.info(f"📝 Total de transcrições encontradas: {len(transcription_files)}")
        for tf in transcription_files:
            logger.info(f"   - {tf}")
        
        return transcription_files
    
    def _copy_transcriptions_to_final(self, transcription_files: List[Path], final_transcriptions_dir: Path, final_audio_name: str) -> List[Dict[str, str]]:
        """
        Copia arquivos de transcrição para o diretório final com nomenclatura padronizada.
        
        Args:
            transcription_files: Lista de arquivos de transcrição
            final_transcriptions_dir: Diretório final para transcrições
            final_audio_name: Nome do áudio final (sem extensão)
            
        Returns:
            Lista de dicionários com informações das transcrições copiadas
        """
        final_transcriptions = []
        
        logger.info(f"📄 Copiando {len(transcription_files)} transcrições para: {final_transcriptions_dir}")
        
        for i, transcription_file in enumerate(transcription_files):
            try:
                # Determinar tipo de STT pelo nome do arquivo ou diretório pai
                if 'whisper' in transcription_file.parent.name.lower():
                    stt_type = 'whisper'
                elif 'wav2vec2' in transcription_file.parent.name.lower():
                    stt_type = 'wav2vec2'
                else:
                    stt_type = 'unknown'
                
                # Nome padronizado para transcrição final
                final_transcription_name = f"{final_audio_name}_{stt_type}.txt"
                final_transcription_path = final_transcriptions_dir / final_transcription_name
                
                logger.info(f"📄 Copiando {stt_type}: {transcription_file.name} → {final_transcription_name}")
                print(f"📄 COPYING TRANSCRIPTION: {transcription_file} → {final_transcription_path}")
                
                # Copiar arquivo
                import shutil
                shutil.copy2(transcription_file, final_transcription_path)
                
                final_transcriptions.append({
                    'type': stt_type,
                    'original_path': str(transcription_file),
                    'final_path': str(final_transcription_path),
                    'filename': final_transcription_name
                })
                
                logger.info(f"✅ Transcrição {stt_type} copiada: {final_transcription_name}")
                
            except Exception as e:
                logger.error(f"❌ Erro ao copiar transcrição {transcription_file}: {e}")
                print(f"❌ TRANSCRIPTION COPY FAILED: {transcription_file} - {e}")
        
        logger.info(f"📄 Total de transcrições copiadas: {len(final_transcriptions)}")
        
        return final_transcriptions


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    pipeline = AudioProcessingPipeline()
    pipeline.cleanup(stages_to_clean=["downloads", "segments", "stt_ready", "audios_abaixo_2,5_MOS", "audios_acima_3,0_MOS", "audios_validados_tts", "audios_denoiser", "clean", "audios_entre_2,5_e_3,0_MOS", "diarization", "overlapping", "speakers"])
