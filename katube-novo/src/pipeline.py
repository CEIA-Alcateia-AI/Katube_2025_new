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
import shutil

from config import Config
from audio_segmenter import AudioSegmenter
from diarizer import EnhancedDiarizer
from overlap_detector import OverlapDetector
from speaker_separator import SpeakerSeparator
from stt_whisper import WhisperSTTTranscriber
from stt_wav2vec2 import WAV2VEC2STTTranscriber
from audio_normalizer import AudioNormalizer
from src.marcos_validation.text_normalizer import process_stt_results as normalize_stt_texts
from denoiser import Denoiser
from sox_normalizer import SoxNormalizer
from mos_filter import MOSQualityFilter

logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """
    Complete pipeline for YouTube audio processing:
    1. Segment audio intelligently
    2. Perform speaker diarization
    3. Detect voice overlaps
    4. Separate audio by speakers
    5. Prepare for STT processing
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

        # Use intelligent segmenter with VAD for quality cuts
        self.segmenter = AudioSegmenter(segment_min_duration, segment_max_duration)
        self.diarizer = EnhancedDiarizer(huggingface_token)
        self.overlap_detector = OverlapDetector()
        self.speaker_separator = SpeakerSeparator()
        
        # Initialize filters
        # Completeness filter moved to separate file (src/audio_completeness_filter.py)
        self.enable_completeness_filter = False  # DISABLED - moved to separate file
        
        logger.info("🔍 Filtros de áudio:")
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
        
        # Initialize STT transcribers (separated models)
        self.enable_stt = True  # Sempre habilitado
        logger.info("🔍 Inicializando STT transcribers separados...")
        
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
        logger.info("✅ Denoiser (DeepFilterNet3) inicializado com sucesso")
        
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
        subdirs = ['segments', 'diarization', 'speakers', 'clean', 'overlapping', 'stt_ready']
        for subdir in subdirs:
            (self.session_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"📁 Session local criada: {self.current_session}")
        logger.info(f"Created session: {self.current_session}")
        return self.session_dir
    


    def _save_timestamps_metadata(self, metadata: Dict[str, Any], update: bool = False):
            """
            Save or update timestamps metadata to JSON file.
            
            Args:
                metadata: Metadata dictionary to save
                update: If True, update existing file; if False, create new
            """
            if not self.session_dir:
                logger.error("Session directory not initialized")
                return
            
            # JSON file location: {session_dir}/segments/segments_timestamps.json
            json_path = self.session_dir / 'segments' / 'segments_timestamps.json'
            
            if update and json_path.exists():
                # Load existing data
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    # Deep merge metadata
                    existing_data.update(metadata)
                    metadata = existing_data
                except Exception as e:
                    logger.warning(f"Failed to load existing metadata: {e}")
            
            # Save metadata
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Timestamps metadata saved to: {json_path}")
            except Exception as e:
                logger.error(f"Failed to save timestamps metadata: {e}")
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
    
    def segment_audio(self, audio_path: Path, use_intelligent_segmentation: bool = True) -> List[Tuple[Path, float, float]]:
        """
        Step 2: Segment audio into manageable chunks for local processing.
        
        Args:
            audio_path: Path to input audio file
            use_intelligent_segmentation: Use intelligent segmentation vs simple chunking
            
        Returns:
            List of tuples (segment_path, absolute_start_time, absolute_end_time)
        """
        logger.info("=== STEP 2: SEGMENTING AUDIO ===")
        
        segments_dir = self.session_dir / 'segments'
        
        if use_intelligent_segmentation:
            # Use intelligent segmentation with VAD for quality cuts - returns timestamps
            segments_with_timestamps = self.segmenter.segment_audio(audio_path, segments_dir)
        else:
            # Simple time-based segmentation fallback
            segments_with_timestamps = self._simple_segment_audio(audio_path, segments_dir)
        
        logger.info(f"Created {len(segments_with_timestamps)} segments")
        
        # Save timestamps metadata to JSON
        self._save_segmentation_metadata(audio_path, segments_with_timestamps)
        
        return segments_with_timestamps


    def _save_segmentation_metadata(self, audio_path: Path, segments_with_timestamps: List[Tuple[Path, float, float]]):
        """
        Save segmentation metadata with absolute timestamps to JSON.
        
        Args:
            audio_path: Path to original audio file
            segments_with_timestamps: List of (segment_path, start_time, end_time) tuples
        """
        import soundfile as sf
        
        # Get audio duration
        try:
            audio_info = sf.info(audio_path)
            total_duration = audio_info.duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            total_duration = 0.0
        
        # Build metadata structure
        metadata = {
            "original_audio": {
                "path": str(audio_path),
                "duration": total_duration
            },
            "segments": {}
        }
        
        # Add each segment info
        for segment_path, start_time, end_time in segments_with_timestamps:
            segment_id = segment_path.stem  # e.g., "segment_000"
            
            metadata["segments"][segment_id] = {
                "file_path": str(segment_path),
                "absolute_start": start_time,
                "absolute_end": end_time,
                "duration": end_time - start_time,
                "speakers": {}  # Will be populated after diarization
            }
        
        # Save to JSON
        self._save_timestamps_metadata(metadata, update=False)
        logger.info(f"Saved segmentation metadata for {len(segments_with_timestamps)} segments")



    def apply_mos_filter(self, segment_paths: List[Path], rejected_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Apply MOS quality filter to audio segments.
        
        Args:
            segment_paths: List of audio segment paths
            rejected_dir: Directory to save rejected segments (optional)
            
        Returns:
            Dictionary with filtering results
        """
        logger.info("=== STEP 4: APPLYING MOS QUALITY FILTER ===")
        
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
        # COLE AQUI - Mudança 3
        # Extrair video_id do primeiro segmento
        video_id = None
        if segment_paths:
            first_segment = segment_paths[0].stem
            video_id = first_segment.split('_segment_')[0] if '_segment_' in first_segment else None

        approved_segments, intermediate_segments, rejected_segments = self.mos_filter.filter_audio_segments(
            segment_paths, 
            output_dir=self.session_dir,
            video_id=video_id
        )
        # For pipeline continuation, use approved segments (≥3.0)
        accepted_segments = approved_segments
        
        # Generate quality report
        quality_report = self.mos_filter.get_quality_report(segment_paths)
        logger.info(f"📊 MOS Quality Report: {quality_report}")
        
        # Log detailed results
        logger.info(f"🎯 MOS filtering results:")
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
        
        logger.info(f"🔍 Filtrando {len(segment_paths)} segmentos por qualidade MOS (OBRIGATÓRIO)...")
        
        # Criar pastas específicas para áudios descartados
        rejected_completeness_dir = self.session_dir / 'audio_descartado_completude'
        rejected_mos_dir = self.session_dir / 'audio_descartado_mos'
        
        # Criar as pastas
        rejected_completeness_dir.mkdir(exist_ok=True)
        rejected_mos_dir.mkdir(exist_ok=True)
        
        # Apply MOS filter with 3-tier classification
        # Extrair video_id do primeiro segmento
        video_id = None
        if segment_paths:
            first_segment = segment_paths[0].stem
            video_id = first_segment.split('_segment_')[0] if '_segment_' in first_segment else None

        approved_segments, intermediate_segments, rejected_segments = self.mos_filter.filter_audio_segments(
            segment_paths,
            output_dir=self.session_dir,
            video_id=video_id
        )
        # For pipeline continuation, use approved segments (≥3.0)
        accepted_segments = approved_segments
        
        # Generate quality report
        quality_report = self.mos_filter.get_quality_report(segment_paths)
        logger.info(f"📊 Relatório de Qualidade MOS: {quality_report}")
        
        return accepted_segments, rejected_segments
        
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
        
        return result
    
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
        Step 5: Separate audio by speakers using diarization results.
        
        Args:
            diarization_results: Results from diarization step
            enhance_audio: Apply audio enhancement
            
        Returns:
            Dictionary with speaker separation results
        """
        logger.info("=== STEP 5: SEPARATING SPEAKERS ===")
        
        speakers_dir = self.session_dir / 'speakers'
        separation_results = {}
        
        # Load timestamps metadata
        json_path = self.session_dir / 'segments' / 'segments_timestamps.json'
        timestamps_metadata = {}
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    timestamps_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load timestamps metadata: {e}")
        
        for audio_path_str, diar_result in diarization_results.items():
            if 'error' in diar_result:
                continue
            
            try:
                audio_path = Path(audio_path_str)
                rttm_path = Path(diar_result['rttm_path']) if diar_result.get('rttm_path') else None
                
                if not rttm_path or not rttm_path.exists():
                    logger.warning(f"No RTTM file for {audio_path.name}")
                    continue
                
                # Get segment offset from metadata
                segment_id = audio_path.stem
                segment_offset = 0.0
                
                if 'segments' in timestamps_metadata and segment_id in timestamps_metadata['segments']:
                    segment_offset = timestamps_metadata['segments'][segment_id].get('absolute_start', 0.0)
                    logger.info(f"Segment {segment_id} offset: {segment_offset:.2f}s")
                
                # Process with speaker separator (with offset)
                result = self.speaker_separator.process_audio_file(
                    audio_path, 
                    rttm_path, 
                    speakers_dir / audio_path.stem,
                    enhance=enhance_audio,
                    create_compilations=True,
                    segment_offset=segment_offset
                )
                
                separation_results[audio_path_str] = result
                
                # Update metadata with speaker information
                self._update_metadata_with_speakers(segment_id, result, rttm_path)
                
            except Exception as e:
                logger.error(f"Speaker separation failed for {audio_path_str}: {e}")
                separation_results[audio_path_str] = {'error': str(e)}
        
        # Summarize results
        total_speakers = sum(r.get('num_speakers', 0) for r in separation_results.values() if 'error' not in r)
        total_segments = sum(r.get('total_segments', 0) for r in separation_results.values() if 'error' not in r)
        
        logger.info(f"Speaker separation: {total_speakers} speakers, {total_segments} segments")
        
        return separation_results
    
    def _update_metadata_with_speakers(self, segment_id: str, separation_result: Dict[str, Any], rttm_path: Path):
        """
        Update timestamps metadata with speaker information after diarization.
        
        Args:
            segment_id: Segment identifier (e.g., "segment_000")
            separation_result: Result from speaker_separator.process_audio_file()
            rttm_path: Path to RTTM file with diarization data
        """
        import pandas as pd
        
        # Load existing metadata
        json_path = self.session_dir / 'segments' / 'segments_timestamps.json'
        if not json_path.exists():
            logger.warning("Timestamps metadata file not found")
            return
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return
        
        # Get segment offset
        segment_offset = 0.0
        if 'segments' in metadata and segment_id in metadata['segments']:
            segment_offset = metadata['segments'][segment_id].get('absolute_start', 0.0)
        else:
            logger.warning(f"Segment {segment_id} not found in metadata")
            return
        
        # Load RTTM to get speaker timestamps
        try:
            speaker_data = self.speaker_separator.load_diarization_dataframe(rttm_path)
            if speaker_data.empty:
                logger.warning(f"No speaker data in RTTM: {rttm_path}")
                return
            
            # Merge consecutive segments
            merged_data = self.speaker_separator.merge_consecutive_segments(speaker_data)
            
        except Exception as e:
            logger.error(f"Failed to load RTTM data: {e}")
            return
        
        # Build speaker information
        speakers_info = {}
        
        for speaker in merged_data['SPEAKER'].unique():
            speaker_segments = merged_data[merged_data['SPEAKER'] == speaker]
            
            segments_list = []
            for idx, row in speaker_segments.iterrows():
                relative_start = row['START']
                relative_end = row['END']
                
                segments_list.append({
                    "relative_start": relative_start,
                    "relative_end": relative_end,
                    "absolute_start": segment_offset + relative_start,
                    "absolute_end": segment_offset + relative_end,
                    "duration": relative_end - relative_start
                })
            
            speakers_info[speaker] = segments_list
        
        # Update metadata
        metadata['segments'][segment_id]['speakers'] = speakers_info
        
        # Save updated metadata
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated metadata with {len(speakers_info)} speakers for segment {segment_id}")
        except Exception as e:
            logger.error(f"Failed to save updated metadata: {e}")

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
        logger.info("=== STEP 7: TRANSCRIBING AUDIO SEGMENTS ===")
        
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
            
            # Step 7.5: Normalize STT texts for validation
            logger.info("=== STEP 7.5: NORMALIZING STT TEXTS ===")
            try:
                normalization_result = normalize_stt_texts(str(self.session_dir))
                
                if normalization_result.get('success'):
                    logger.info(f"Text normalization completed:")
                    logger.info(f"   - Total videos: {normalization_result.get('total_videos', 0)}")
                    logger.info(f"   - Total segments: {normalization_result.get('total_segments', 0)}")
                    logger.info(f"   - Output files: {len(normalization_result.get('output_files', []))}")
                    
                    combined_results['normalization'] = normalization_result
                else:
                    logger.warning(f"Text normalization failed: {normalization_result.get('error')}")
                    combined_results['normalization'] = {"error": normalization_result.get('error')}
                    
            except Exception as e:
                logger.error(f"Error in text normalization: {e}")
                combined_results['normalization'] = {"error": str(e)}

            # Step 8: Validate STT transcriptions with Levenshtein + MOS
            logger.info("=== STEP 8: VALIDATING STT TRANSCRIPTIONS ===")
            if normalization_result.get('success'):
                try:
                    # Importar funcao de validacao
                    from src.marcos_validation.validador_transcricao import validate_normalized_texts
                    
                    # Buscar arquivo JSON normalizado
                    normalized_json_path = normalization_result.get('output_file')
                    
                    if normalized_json_path:
                        # Executar validacao (Levenshtein + MOS)
                        validation_result = validate_normalized_texts(normalized_json_path)
                        
                        if validation_result.get('success'):
                            combined_results['validation'] = validation_result
                            logger.info(f"STT validation completed:")
                            logger.info(f"   - Average similarity: {validation_result.get('average_similarity', 0):.3f}")
                            logger.info(f"   - MOS scores found: {validation_result.get('mos_scores_found', 0)}/{validation_result.get('validated_segments', 0)}")
                            logger.info(f"   - Output file: {validation_result.get('output_file')}")
                        else:
                            logger.warning(f"Text validation failed: {validation_result.get('error')}")
                            combined_results['validation'] = {"error": validation_result.get('error')}
                    else:
                        logger.warning("Normalization output file not found, skipping validation")
                        combined_results['validation'] = {"error": "No normalized file to validate"}
                        
                except Exception as e:
                    logger.error(f"Error in text validation: {e}")
                    combined_results['validation'] = {"error": str(e)}
            else:
                logger.warning("Skipping validation - normalization failed")
                combined_results['validation'] = {"error": "Normalization failed"}
            

            # Step 9: Filter by similarity threshold and MOS, then apply denoising
            if combined_results.get('validation', {}).get('success'):
                try:
                    validation_json_path = combined_results['validation'].get('output_file')
                    
                    if validation_json_path:
                        filter_result = self.filter_and_denoise_segments(
                            validation_json_path=validation_json_path,
                            output_dir=self.session_dir,
                            similarity_threshold=0.80,
                            mos_range=(2.5, 3.0)
                        )
                        combined_results['filter_and_denoise'] = filter_result
                        logger.info(f"Filtering and denoising completed:")
                        logger.info(f"   - Approved count: {filter_result.get('approved_count', 0)}")
                        logger.info(f"   - Denoised count: {filter_result.get('denoised_success', 0)}")
                        
                        # Step 10: Sox normalization of approved audios
                        if filter_result.get('success'):
                            try:
                                denoiser_dir = Path(filter_result.get('denoiser_dir'))
                                video_id = filter_result.get('video_id')
                                
                                sox_result = self.sox_normalize_approved_audios(
                                    denoiser_dir=denoiser_dir,
                                    video_id=video_id
                                )
                                combined_results['sox_normalization'] = sox_result
                                
                                if sox_result.get('success'):
                                    logger.info(f"Sox normalization completed:")
                                    logger.info(f"   - Normalized: {sox_result.get('success_count', 0)}/{sox_result.get('total_files', 0)}")
                                    logger.info(f"   - Output: {sox_result.get('output_dir')}")
                                else:
                                    logger.warning(f"Sox normalization failed: {sox_result.get('error')}")
                                    
                            except Exception as e:
                                logger.error(f"Error in Sox normalization: {e}")
                                combined_results['sox_normalization'] = {"error": str(e)}
                    else:
                        logger.warning("Validation output file not found, skipping filter and denoise")
                        
                except Exception as e:
                    logger.error(f"Error in filter and denoise: {e}")
                    combined_results['filter_and_denoise'] = {"error": str(e)}
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in transcription step: {e}")
            return {"error": str(e)}
    
    def process_local_audio(self,  
                           audio_path: Path, 
                           num_speakers: Optional[int] = None,
                           enhance_audio: bool = True,
                           use_intelligent_segmentation: bool = True,
                           session_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: process a audio through all steps.
        
        Args:
            audio_path: audio to segment and process
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
        logger.info(f"Processing file: {audio_path}")
        
        #Validação do áudio de entrada
        if not audio_path.exists():
            logger.error(f"❌ Audio directory does not exist: {audio_path}")
            return {'success': False, 'error': f"Audio file does not exist: {audio_path}"}
        
        flac_files = list(audio_path.glob('*.flac'))
        if not flac_files:
            error_msg = f"Nenhum arquivo .flac encontrado no diretório: {audio_path}"
            logger.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        source_audio_path = flac_files[0]

        try:
            # Create session
            session_name_resolved = session_name or audio_path.stem
            session_dir = self.create_session(session_name_resolved)
            
            # Step 1: Segment
            segments_with_timestamps = self.segment_audio(source_audio_path, use_intelligent_segmentation)
            
            # Extract paths for processing (filters expect List[Path])
            segments = [seg_path for seg_path, _, _ in segments_with_timestamps]
            
            # Step 2: Apply completeness filter (DISABLED - moved to separate file)
            # Completeness filter is now in src/audio_completeness_filter.py
            # if self.enable_completeness_filter:
            #     completeness_rejected_dir = session_dir / 'audio_descartado_completude'
            #     completeness_result = self.apply_completeness_filter(segments, rejected_dir=completeness_rejected_dir)
            #     segments = completeness_result['complete_segments']
            #     logger.info(f"Completeness filter: {len(segments)} segments passed (filtered {completeness_result['cut_count']} cut segments)")
            
            # Step 3: Apply MOS filter
            if self.enable_mos_filter:
                try:
                    mos_rejected_dir = session_dir / 'audio_descartado_mos'
                    mos_result = self.apply_mos_filter(segments, rejected_dir=mos_rejected_dir)
                    segments = mos_result['filtered_segments']
                    logger.info(f"MOS filter: {len(segments)} segments passed")
                except Exception as e:
                    logger.error(f"MOS filter failed: {e}")
                    return {'success': False, 'error': f"MOS filter failed: {str(e)}"}
            
            # Step 4: Diarization (ANTES do STT)
            diarization_results = self.perform_diarization(segments, num_speakers)
            
            # Step 5: Overlap detection (ANTES do STT)
            clean_segments, overlapping_segments = self.detect_overlaps(segments)
            
            # Step 6: Speaker separation (ANTES do STT)
            separation_results = self.separate_speakers(diarization_results, enhance_audio)
            
            # Step 7: STT preparation (ANTES do STT)
            stt_files = self.prepare_for_stt(separation_results)
            
            # Step 8: Apply STT transcription
            stt_result = {}
            if self.enable_stt:
                try:
                    # Flatten stt_files dictionary to list of paths
                    segment_paths = []
                    if stt_files:
                        for speaker_files in stt_files.values():
                            segment_paths.extend(speaker_files)
                    else:
                        segment_paths = segments

                    stt_result = self.transcribe_audio_segments(segment_paths)
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
            
            # Step 09: Move approved segments to final directory
            try:
                final_segments_dir = session_dir / 'segments_aprovados'
                final_segments_dir.mkdir(exist_ok=True)
                                
                final_segments = []
                import shutil

                # Flatten stt_files dictionary to list if needed
                if stt_files:
                    segments_to_move = []
                    for speaker_files in stt_files.values():
                        segments_to_move.extend(speaker_files)
                else:
                    segments_to_move = segments

                for segment in segments_to_move:
                    if isinstance(segment, Path):
                        final_path = final_segments_dir / segment.name
                        shutil.copy2(segment, final_path)
                        final_segments.append(final_path)
                
                segments = final_segments
                logger.info(f"✅ {len(segments)} segments moved to final approved directory")
            except Exception as e:
                logger.warning(f"⚠️ Could not move segments to final directory: {e}")
            
            # Final results
            processing_time = time.time() - start_time
            
            results = {
                'session_name': self.current_session,
                'session_dir': str(session_dir),
                'processing_time': processing_time,
                'downloaded_audio': str(audio_path),
                'num_segments': len(segments),
                'num_clean_segments': len(clean_segments),
                'num_overlapping_segments': len(overlapping_segments),
                'diarization_results': diarization_results,
                'separation_results': separation_results,
                'stt_ready_files': stt_files,
                'stt_results': stt_result,  # Include STT validation and filtering results
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
            
            logger.info("===\n\n\n LIMPEZA DE DIRETÓRIOS INTERMEDIÁRIOS ===")
            #self.cleanup(stages_to_clean=["downloads", "audio_rejeitado_validacao","segments", "stt_ready","stt_results\STT-wav2vec2", "stt_results\STT-whisper", "audios_abaixo_2,5_MOS", "audios_acima_3,0_MOS", "audios_validados_tts", "audios_denoiser", "clean", "audios_entre_2,5_e_3,0_MOS", "diarization", "overlapping", "speakers"])

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
        logger.info("=== STEP 8: VALIDATING STT TRANSCRIPTIONS ===")
        
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
                                   validation_json_path: str,
                                   output_dir: Path,
                                   similarity_threshold: float = 0.80,
                                   mos_range: Tuple[float, float] = (2.5, 3.0)) -> Dict[str, Any]:
        """
        Step 9: Filter segments by similarity AND MOS, apply denoising when needed.
        Creates final JSON with all segment data + utilizou_denoiser status.
        
        Args:
            validation_json_path: Path to validation JSON file
            output_dir: Session directory
            similarity_threshold: Minimum similarity to accept (default 0.80)
            mos_range: MOS range for denoising (default (2.5, 3.0))
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"=== STEP 9: FILTERING AND DENOISING SEGMENTS ===")
        logger.info(f"Similarity threshold: >= {similarity_threshold}")
        logger.info(f"MOS range for denoising: [{mos_range[0]}, {mos_range[1]}]")
        
        try:
            import json
            import shutil
            from pathlib import Path as PathLib
            
            # Carregar JSON de validacao
            validation_path = PathLib(validation_json_path)
            
            if not validation_path.exists():
                error_msg = f"Validation JSON not found: {validation_json_path}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            with open(validation_path, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
            
            video_id = validation_data.get('video_id', 'unknown')
            normalized_pairs = validation_data.get('normalized_pairs', {})
            
            if not normalized_pairs:
                error_msg = "No normalized pairs found in validation JSON"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            logger.info(f"Processing {len(normalized_pairs)} segments for video: {video_id}")
            
            # Criar pasta de saida (todos os aprovados ficam aqui)
            denoiser_dir = output_dir / 'audios_denoiser'
            denoiser_dir.mkdir(parents=True, exist_ok=True)
            
            # Criar pastas de rejeitados
            rejected_similarity_dir = output_dir / 'audio_rejeitado_validacao'
            rejected_mos_dir = output_dir / 'audio_rejeitado_mos_range'
            rejected_similarity_dir.mkdir(parents=True, exist_ok=True)
            rejected_mos_dir.mkdir(parents=True, exist_ok=True)
            
            # Contadores para estatisticas
            approved_with_denoise = 0
            approved_without_denoise = 0
            rejected_by_similarity = 0
            rejected_by_mos = 0
            denoised_success = 0
            
            # JSON final que sera salvo
            final_json = {
                "video_id": video_id,
                "total_segments": len(normalized_pairs),
                "approved_count": 0,
                "rejected_count": 0,
                "segments": {}
            }
            
            # Processar cada segmento
            for segment_id, pair_data in normalized_pairs.items():
                similarity = pair_data.get('levenshtein_similarity', 0.0)
                mos_score = pair_data.get('mos_score')
                flac_file = pair_data.get('flac_file')
                
                logger.info(f"\nProcessing: {segment_id}")
                logger.info(f"  Similarity: {similarity:.3f}, MOS: {mos_score}")
                
                # Inicializar campos novos
                utilizou_denoiser = False
                status = None
                
                # Buscar arquivo FLAC original
                audio_file = None
                stt_ready_dir = output_dir / 'stt_ready'
                
                if stt_ready_dir.exists():
                    for speaker_dir in stt_ready_dir.iterdir():
                        if speaker_dir.is_dir():
                            for flac_path in speaker_dir.glob("*.flac"):
                                # Extrair prefixo do segment_id (antes de _stt_)
                                prefix = segment_id.split('_stt_')[0] if '_stt_' in segment_id else segment_id
                                if prefix in flac_path.stem or (flac_file and flac_file in flac_path.name):
                                    audio_file = flac_path
                                    break
                        if audio_file:
                            break
                
                if not audio_file or not audio_file.exists():
                    logger.warning(f"  Audio file not found, skipping")
                    continue
                
                # FILTRO 1: Similaridade
                if similarity < similarity_threshold:
                    status = "rejected_similarity"
                    utilizou_denoiser = False
                    rejected_by_similarity += 1
                    
                    logger.info(f"  REJECTED by similarity ({similarity:.3f} < {similarity_threshold})")
                    
                    # Copiar para pasta de rejeitados
                    rejected_path = rejected_similarity_dir / f"{segment_id}.flac"
                    shutil.copy2(audio_file, rejected_path)
                
                # FILTRO 2: MOS (se passou pelo filtro de similaridade)
                elif mos_score is None:
                    logger.warning(f"  No MOS score available, skipping")
                    continue
                
                elif mos_score < mos_range[0]:
                    status = "rejected_mos"
                    utilizou_denoiser = False
                    rejected_by_mos += 1
                    
                    logger.info(f"  REJECTED by MOS ({mos_score} < {mos_range[0]})")
                    
                    # Copiar para pasta de rejeitados por MOS
                    rejected_path = rejected_mos_dir / f"{segment_id}.flac"
                    shutil.copy2(audio_file, rejected_path)
                
                # APROVADO
                else:
                    status = "approved"
                    
                    # Decidir se aplica denoiser
                    if mos_range[0] <= mos_score <= mos_range[1]:
                        # Precisa denoising
                        utilizou_denoiser = True
                        approved_with_denoise += 1
                        
                        logger.info(f"  APPROVED - will apply DENOISER (MOS in [{mos_range[0]}, {mos_range[1]}])")
                        
                        # Aplicar denoising
                        denoised_path = denoiser_dir / f"{segment_id}.flac"
                        
                        try:
                            logger.info(f"  Applying DeepFilterNet3...")
                            self.denoiser.process_file(
                                str(audio_file),
                                str(denoised_path)
                            )
                            denoised_success += 1
                            logger.info(f"  Denoised successfully")
                        except Exception as e:
                            logger.error(f"  Error during denoising: {e}")
                            # Se falhar, copiar original
                            shutil.copy2(audio_file, denoised_path)
                    
                    else:
                        # MOS > 3.0 - audio excelente, nao precisa denoising
                        utilizou_denoiser = False
                        approved_without_denoise += 1
                        
                        logger.info(f"  APPROVED - NO denoising needed (MOS {mos_score} > {mos_range[1]})")
                        
                        # Copiar original para pasta final
                        approved_path = denoiser_dir / f"{segment_id}.flac"
                        shutil.copy2(audio_file, approved_path)
                        logger.info(f"  Copied original to audios_denoiser/")
                
                # Adicionar ao JSON final (TODOS os segmentos, aprovados e rejeitados)
                final_json["segments"][segment_id] = {
                    "txt_whisper": pair_data.get('txt_whisper'),
                    "txt_wav2vec2": pair_data.get('txt_wav2vec2'),
                    "flac_file": flac_file,
                    "whisper_original": pair_data.get('whisper_original'),
                    "whisper_normalized": pair_data.get('whisper_normalized'),
                    "wav2vec2_original": pair_data.get('wav2vec2_original'),
                    "wav2vec2_normalized": pair_data.get('wav2vec2_normalized'),
                    "levenshtein_similarity": similarity,
                    "mos_score": mos_score,
                    "utilizou_denoiser": utilizou_denoiser,
                    "status": status
                }
            
            # Calcular estatisticas finais
            approved_total = approved_with_denoise + approved_without_denoise
            rejected_total = rejected_by_similarity + rejected_by_mos
            
            final_json["approved_count"] = approved_total
            final_json["rejected_count"] = rejected_total
            
            # Salvar JSON final
            final_json_path = denoiser_dir / f"{video_id}_final_audio_dataset.json"
            
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"FILTER AND DENOISE COMPLETED")
            logger.info(f"{'='*60}")
            logger.info(f"Total segments processed: {len(normalized_pairs)}")
            logger.info(f"\nAPPROVED: {approved_total}")
            logger.info(f"  - With denoising: {approved_with_denoise} (successfully denoised: {denoised_success})")
            logger.info(f"  - Without denoising (MOS > {mos_range[1]}): {approved_without_denoise}")
            logger.info(f"\nREJECTED: {rejected_total}")
            logger.info(f"  - By similarity: {rejected_by_similarity}")
            logger.info(f"  - By MOS: {rejected_by_mos}")
            logger.info(f"\nFinal JSON saved: {final_json_path}")
            logger.info(f"All approved audio files in: {denoiser_dir}")
            logger.info(f"{'='*60}\n")
            
            return {
                'success': True,
                'video_id': video_id,
                'total_segments': len(normalized_pairs),
                'approved_count': approved_total,
                'approved_with_denoise': approved_with_denoise,
                'approved_without_denoise': approved_without_denoise,
                'rejected_count': rejected_total,
                'rejected_by_similarity': rejected_by_similarity,
                'rejected_by_mos': rejected_by_mos,
                'denoised_success': denoised_success,
                'final_json_path': str(final_json_path),
                'denoiser_dir': str(denoiser_dir)
            }
            
        except Exception as e:
            logger.error(f"Error in filter_and_denoise_segments: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def sox_normalize_approved_audios(self, 
                                      denoiser_dir: Path,
                                      video_id: str) -> Dict[str, Any]:
        """
        Step 10: Normalize approved audio files with Sox.
        Saves to: /katube-novo/dataset/audio_dataset/{video_id}/
        
        Args:
            denoiser_dir: Directory with approved audio files (audios_denoiser/)
            video_id: Video ID for folder name
            
        Returns:
            Dictionary with normalization results
        """
        logger.info(f"=== STEP 10: SOX NORMALIZATION ===")
        
        try:
            # Caminho base do dataset
            dataset_base = Path("/home/anjos/Dropbox/PROJETO CEIA/Alcateia/Katube_2025_new/katube-novo/dataset/audio_dataset")
            output_dir = dataset_base / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Dataset base: {dataset_base}")
            logger.info(f"Output directory: {output_dir}")
            
            # Buscar todos os audios aprovados (exceto JSON)
            audio_files = [f for f in denoiser_dir.glob("*.flac")]
            
            if not audio_files:
                logger.warning("No audio files found in denoiser directory")
                return {"success": False, "error": "No audio files to normalize"}
            
            logger.info(f"Found {len(audio_files)} audio files to normalize")
            
            # Contadores
            success_count = 0
            failed_count = 0
            normalized_files = []
            
            # Normalizar cada audio
            for i, audio_file in enumerate(audio_files, 1):
                logger.info(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
                
                # Output mantém nome original
                output_path = output_dir / audio_file.name
                
                # Normalizar com Sox
                result = self.sox_normalizer.normalize_audio(audio_file, output_path)
                
                if result['success']:
                    success_count += 1
                    normalized_files.append(str(output_path))
                    logger.info(f"  SUCCESS -> {output_path.name}")
                else:
                    failed_count += 1
                    logger.error(f"  FAILED: {result.get('error')}")
            
            # Log final
            logger.info(f"\n{'='*60}")
            logger.info(f"SOX NORMALIZATION COMPLETED")
            logger.info(f"{'='*60}")
            logger.info(f"Total files: {len(audio_files)}")
            logger.info(f"  Success: {success_count}")
            logger.info(f"  Failed: {failed_count}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"{'='*60}\n")
            
            return {
                'success': True,
                'video_id': video_id,
                'total_files': len(audio_files),
                'success_count': success_count,
                'failed_count': failed_count,
                'output_dir': str(output_dir),
                'normalized_files': normalized_files
            }
            
        except Exception as e:
            logger.error(f"Error in Sox normalization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

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
            """Recursively convert Path objects and Annotation objects to strings for JSON serialization."""
            from pyannote.core import Annotation
            import pandas as pd
            
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Annotation):
                # Convert Annotation to string representation or skip it
                return str(obj)
            elif isinstance(obj, pd.DataFrame):
                # Converte DataFrame para lista de dicionarios
                return obj.to_dict(orient='records')
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
    
    # Example: process a audio file
    results = pipeline.process_local_audio(
         r"C:\Igor\BIA\Alcateia\Katube_2025_new\katube-novo\BZ-QBv4Vc5k_chunk_00.flac"
    )
    print(f"Pipeline results: {results['statistics']}")
