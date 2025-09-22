#!/usr/bin/env python3
"""
Flask web interface for YouTube Audio Processing Pipeline
"""
# Suppress warnings first
from src.warning_suppression import *

import os
import sys
import json
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pipeline import AudioProcessingPipeline
from src.config import Config

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Global variables for job tracking
active_jobs = {}
job_lock = threading.Lock()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus:
    def __init__(self, job_id: str, url: str):
        self.job_id = job_id
        self.url = url
        self.status = "waiting"  # waiting, downloading, segmenting, diarizing, separating, completed, failed
        self.progress = 0  # 0-100
        self.message = "Iniciando processamento..."
        self.start_time = datetime.now()
        self.end_time = None
        self.results = None
        self.error = None
        
    def update(self, status: str, progress: int, message: str):
        self.status = status
        self.progress = progress
        self.message = message
        logger.info(f"Job {self.job_id}: {status} - {progress}% - {message}")
        
    def complete(self, results: dict):
        self.status = "completed"
        self.progress = 100
        self.message = "Processamento concluído com sucesso!"
        self.end_time = datetime.now()
        self.results = results
        
    def fail(self, error: str):
        self.status = "failed"
        self.progress = 0
        self.message = f"Erro: {error}"
        self.end_time = datetime.now()
        self.error = error

def process_youtube_url_background(job_id: str, url: str, options: dict):
    """Background task to process YouTube URL"""
    job = active_jobs[job_id]
    
    try:
        # Create pipeline
        pipeline = AudioProcessingPipeline(
            output_base_dir=Config.OUTPUT_DIR,
            huggingface_token=os.getenv('HUGGINGFACE_TOKEN'),
            segment_min_duration=options.get('min_duration', 10.0),
            segment_max_duration=options.get('max_duration', 15.0)
        )
        
        # Update job status throughout the process
        job.update("downloading", 10, "Baixando áudio do YouTube...")
        
        # Create session
        session_name = options.get('session_name') or f"web_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = pipeline.create_session(session_name)
        
        # Download
        audio_path = pipeline.download_youtube_audio(url, options.get('filename'))
        job.update("normalizing", 20, "Normalizando áudio (FLAC, 24kHz, Mono)...")
        job.update("segmenting", 25, "Segmentando áudio...")
        
        # Segment
        segments = pipeline.segment_audio(audio_path, options.get('intelligent_segmentation', True))
        job.update("filtering", 30, "Aplicando filtros de qualidade...")
        
        # Apply completeness filter (DISABLED - moved to separate file)
        # Completeness filter is now in src/audio_completeness_filter.py
        # if pipeline.enable_completeness_filter:
        #     completeness_rejected_dir = session_dir / 'audio_descartado_completude'
        #     completeness_result = pipeline.apply_completeness_filter(segments, rejected_dir=completeness_rejected_dir)
        #     segments = completeness_result['complete_segments']
        #     job.update("filtering", 35, f"Filtro completude: {len(segments)} segmentos aprovados")
        
        # Apply MOS filter
        if pipeline.enable_mos_filter:
            mos_rejected_dir = session_dir / 'audio_descartado_mos'
            mos_result = pipeline.apply_mos_filter(segments, rejected_dir=mos_rejected_dir)
            segments = mos_result['filtered_segments']
            job.update("filtering", 40, f"Filtro MOS: {len(segments)} segmentos aprovados")
            
            # Move approved segments to segments_aprovados immediately after MOS filter
            segments_aprovados_dir = session_dir / 'segments_aprovados'
            segments_aprovados_dir.mkdir(exist_ok=True)
            
            approved_segments = []
            for segment in segments:
                approved_path = segments_aprovados_dir / segment.name
                import shutil
                shutil.copy2(segment, approved_path)
                approved_segments.append(approved_path)
            
            # Update segments to point to approved location
            segments = approved_segments
        
        # Diarization (ANTES do STT)
        job.update("diarizing", 50, "Executando diarização...")
        diarization_results = pipeline.perform_diarization(segments, options.get('num_speakers'))
        
        # Overlap detection (ANTES do STT)
        job.update("overlap", 60, "Detectando sobreposições...")
        clean_segments, overlapping_segments = pipeline.detect_overlaps(segments)
        
        # Speaker separation (ANTES do STT)
        job.update("separating", 70, "Separando por locutor...")
        separation_results = pipeline.separate_speakers(diarization_results, options.get('enhance_audio', True))
        
        # STT preparation (ANTES do STT)
        job.update("preparing", 75, "Preparando arquivos para STT...")
        stt_files = pipeline.prepare_for_stt(separation_results)
        
        # Apply STT transcription
        if pipeline.enable_stt:
            # Convert stt_files dict to a flat list of all files
            if stt_files:
                all_stt_files = []
                for speaker_files in stt_files.values():
                    all_stt_files.extend(speaker_files)
                stt_result = pipeline.transcribe_audio_segments(all_stt_files)
            else:
                stt_result = pipeline.transcribe_audio_segments(segments)
            validation_info = ""
            filter_info = ""
            
            if 'validation' in stt_result and stt_result['validation'].get('success'):
                avg_sim = stt_result['validation'].get('average_similarity', 0)
                validation_info = f" (Validação: {avg_sim:.3f} similaridade)"
            
            if 'filter_and_denoise' in stt_result and stt_result['filter_and_denoise'].get('success'):
                validated_count = stt_result['filter_and_denoise'].get('validated_count', 0)
                denoised_count = stt_result['filter_and_denoise'].get('denoised_count', 0)
                filter_info = f" | Filtro 80%: {validated_count} validados, {denoised_count} denoised"
            
            job.update("filtering", 85, f"STT: {stt_result.get('whisper_count', 0)} Whisper + {stt_result.get('wav2vec2_count', 0)} WAV2VEC2{validation_info}{filter_info}")
        
        # Complete results
        processing_time = time.time() - job.start_time.timestamp()
        
        results = {
            'session_name': session_name,
            'session_dir': str(session_dir),
            'url': url,
            'processing_time': processing_time,
            'downloaded_audio': str(audio_path),
            'num_segments': len(segments),
            'num_clean_segments': len(clean_segments) if 'clean_segments' in locals() else 0,
            'num_overlapping_segments': len(overlapping_segments) if 'overlapping_segments' in locals() else 0,
            'diarization_results': diarization_results if 'diarization_results' in locals() else {},
            'separation_results': separation_results if 'separation_results' in locals() else {},
            'stt_ready_files': stt_files if 'stt_files' in locals() else [],
            'stt_results': stt_result if 'stt_result' in locals() else {}
        }
        
        # Save results to JSON
        results_file = session_dir / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            # Simple JSON serialization for results
            import json
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        job.complete(results)
        
    except Exception as e:
        logger.error(f"Background job {job_id} failed: {e}")
        job.fail(str(e))

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/process_channel', methods=['POST'])
def process_channel():
    """Process entire YouTube channel."""
    try:
        data = request.get_json()
        channel_url = data.get('channelUrl') or data.get('channel_url')
        
        # Validate URL (accept both channel and video URLs)
        if not channel_url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Check if it's a channel URL or video URL
        is_channel = any(pattern in channel_url.lower() for pattern in [
            '/channel/', '/c/', '/user/', '/@', '/playlist?'
        ])
        
        if not is_channel and 'youtube.com/watch' not in channel_url.lower() and 'youtu.be/' not in channel_url.lower():
            return jsonify({'error': 'Please provide a valid YouTube channel or video URL'}), 400
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Start background job using threading
        job_thread = threading.Thread(
            target=process_channel_background,
            args=(job_id, channel_url)
        )
        job_thread.daemon = True
        job_thread.start()
        
        # Store job status
        with job_lock:
            active_jobs[job_id] = JobStatus(job_id, channel_url)
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Channel processing started'
        })
        
    except Exception as e:
        logger.error(f"Error starting channel processing: {e}")
        return jsonify({'error': str(e)}), 500

def process_channel_background(job_id, url):
    """Background job to process YouTube channel or single video."""
    try:
        logger.info(f"🔄 Starting processing: {url}")
        
        # Check if it's a channel URL or video URL
        is_channel = any(pattern in url.lower() for pattern in [
            '/channel/', '/c/', '/user/', '/@', '/playlist?'
        ])
        
        # Update job status
        with job_lock:
            if job_id in active_jobs:
                if is_channel:
                    active_jobs[job_id].update('downloading', 10, 'Iniciando processamento do canal...')
                else:
                    active_jobs[job_id].update('downloading', 10, 'Iniciando processamento do vídeo...')
        
        # Initialize pipeline with MOS filter (OBRIGATÓRIO)
        pipeline = AudioProcessingPipeline(
            huggingface_token=os.getenv('HUGGINGFACE_TOKEN'),
            mos_threshold=2.0
        )
        
        processed_count = 0
        failed_count = 0
        
        def progress_callback(video_url, success, total_videos, current_index):
            """Update progress for each video processed."""
            nonlocal processed_count, failed_count
            
            with job_lock:
                if job_id in active_jobs:
                    job = active_jobs[job_id]
                    
                    # Always count as processed (success) since we're processing all videos
                    processed_count += 1
                    
                    # Extract video ID for display
                    video_id = video_url.split('watch?v=')[-1].split('&')[0] if 'watch?v=' in video_url else video_url[-11:]
                    
                    # Show progress as "Video X of Y" format
                    status = f"📹 Vídeo {current_index}/{total_videos} processado"
                    
                    # Calculate progress percentage based on actual total
                    progress_percent = min(90, int((current_index / total_videos) * 90))
                    
                    job.update('processing', progress_percent, 
                             f'Processando canal... {status} | ID: {video_id}')
        
        # Process based on URL type
        if is_channel:
            # Process entire channel
            result = pipeline.process_youtube_channel(url, max_videos=2500, progress_callback=progress_callback)
            
            # Update final job status
            with job_lock:
                if job_id in active_jobs:
                    if result.get('success', False):
                        total_videos = result.get('total_videos', 0)
                        processed = result.get('videos_processed', 0)
                        failed = result.get('videos_failed', 0)
                        active_jobs[job_id].update('completed', 100, f'✅ Canal processado! {processed}/{total_videos} vídeos')
                    else:
                        active_jobs[job_id].update('failed', 0, f'❌ Erro no canal: {result.get("error", "Erro desconhecido")}')
                    active_jobs[job_id].results = result
            
            logger.info(f"✅ Channel processing complete: {result}")
        else:
            # Process single video
            result = pipeline.process_single_video(url)
            
            # Update final job status
            with job_lock:
                if job_id in active_jobs:
                    if result.get('success', False):
                        segments_count = result.get('segments_count', 0)
                        speakers_count = result.get('speakers_count', 0)
                        active_jobs[job_id].update('completed', 100, f'✅ Vídeo processado! {segments_count} segmentos, {speakers_count} falantes')
                    else:
                        active_jobs[job_id].update('failed', 0, f'❌ Erro no vídeo: {result.get("error", "Erro desconhecido")}')
                    active_jobs[job_id].results = result
            
            logger.info(f"✅ Video processing complete: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Channel processing failed: {e}")
        
        # Update job status with error
        with job_lock:
            if job_id in active_jobs:
                active_jobs[job_id].update('failed', 0, f'Erro: {str(e)}')
                active_jobs[job_id].error = str(e)
        
        return {'success': False, 'error': str(e)}

@app.route('/process', methods=['POST'])
def process_url():
    """Start processing a YouTube URL"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL é obrigatória'}), 400
        
        # Validate YouTube URL
        if 'youtube.com/watch' not in url and 'youtu.be/' not in url:
            return jsonify({'error': 'URL inválida. Use uma URL válida do YouTube.'}), 400
        
        # Create job
        job_id = str(uuid.uuid4())
        
        # Get options from request
        options = {
            'filename': data.get('filename'),
            'num_speakers': data.get('num_speakers'),
            'min_duration': data.get('min_duration', 10.0),
            'max_duration': data.get('max_duration', 15.0),
            'enhance_audio': data.get('enhance_audio', True),
            'intelligent_segmentation': data.get('intelligent_segmentation', True),
            'session_name': data.get('session_name')
        }
        
        # Create job status
        with job_lock:
            active_jobs[job_id] = JobStatus(job_id, url)
        
        # Start background processing
        thread = threading.Thread(
            target=process_youtube_url_background,
            args=(job_id, url, options),
            daemon=True
        )
        thread.start()
        
        return jsonify({'job_id': job_id})
        
    except Exception as e:
        logger.error(f"Process URL error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    with job_lock:
        job = active_jobs.get(job_id)
        
        if not job:
            return jsonify({'error': 'Job não encontrado'}), 404
        
        return jsonify({
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'message': job.message,
            'start_time': job.start_time.isoformat(),
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'error': job.error
        })

@app.route('/result/<job_id>')
def get_result(job_id):
    """Get job results"""
    with job_lock:
        job = active_jobs.get(job_id)
        
        if not job:
            return jsonify({'error': 'Job não encontrado'}), 404
        
        if job.status != 'completed':
            return jsonify({'error': 'Job ainda não foi concluído'}), 400
        
        # Clean results for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__module__') and 'pyannote' in str(obj.__module__):
                return f"pyannote.{obj.__class__.__name__}"  # Handle pyannote objects
            elif hasattr(obj, '__dict__'):
                return str(obj)  # Convert complex objects to string
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif hasattr(obj, '__class__'):
                return f"{obj.__class__.__module__}.{obj.__class__.__name__}"  # Better class representation
            else:
                return str(obj)  # Convert anything else to string
        
        return jsonify({
            'job_id': job.job_id,
            'status': job.status,
            'results': clean_for_json(job.results),
            'processing_time': (job.end_time - job.start_time).total_seconds() if job.end_time and job.start_time else 0
        })

@app.route('/results/<job_id>')
def results_page(job_id):
    """Results page"""
    with job_lock:
        job = active_jobs.get(job_id)
        
        if not job:
            return "Job não encontrado", 404
            
    return render_template('result.html', job_id=job_id)

@app.route('/download/<job_id>/<path:file_type>')
def download_file(job_id, file_type):
    """Download processed files"""
    with job_lock:
        job = active_jobs.get(job_id)
        
        if not job or job.status != 'completed':
            return "Arquivo não disponível", 404
        
        session_dir = Path(job.results['session_dir'])
        
        try:
            if file_type == 'results.json':
                file_path = session_dir / 'pipeline_results.json'
                return send_file(file_path, as_attachment=True, download_name=f'results_{job_id}.json')
            
            elif file_type.startswith('speaker_'):
                # Download specific speaker files as ZIP
                import zipfile
                import tempfile
                
                speaker_id = file_type.replace('speaker_', '')
                speaker_dir = session_dir / 'stt_ready' / f'speaker_{speaker_id}'
                
                if not speaker_dir.exists():
                    return "Speaker não encontrado", 404
                
                # Create temporary ZIP file
                temp_zip = tempfile.mktemp(suffix='.zip')
                
                with zipfile.ZipFile(temp_zip, 'w') as zipf:
                    for file_path in speaker_dir.glob('*'):
                        if file_path.is_file():
                            zipf.write(file_path, file_path.name)
                
                return send_file(temp_zip, as_attachment=True, download_name=f'speaker_{speaker_id}_{job_id}.zip')
            
            else:
                return "Tipo de arquivo inválido", 400
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            return "Erro no download", 500

@app.route('/cleanup/<job_id>', methods=['POST'])
def cleanup_job(job_id):
    """Clean up job data"""
    with job_lock:
        if job_id in active_jobs:
            del active_jobs[job_id]
    
    return jsonify({'message': 'Job removido'})

@app.route('/jobs')
def list_jobs():
    """List all active jobs (for debugging)"""
    with job_lock:
        jobs_info = []
        for job_id, job in active_jobs.items():
            jobs_info.append({
                'job_id': job_id,
                'status': job.status,
                'progress': job.progress,
                'url': job.url,
                'start_time': job.start_time.isoformat()
            })
    
    return jsonify(jobs_info)

if __name__ == '__main__':
    
    # Ensure directories exist
    Config.create_directories()
    
    # Create additional directories
    Config.AUDIOS_BAIXADOS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 YouTube Audio Processing Pipeline - Web Interface")
    print("============================================================")
    print("📁 Processamento local - Arquivos salvos no disco")
    print("🎯 Aceita: Canais YouTube OU vídeos individuais")
    print("🔍 Pipeline: Download → Normalização → Segmentação → MOS → Diarização → Overlap → Separação → STT → Validação → Denoiser")
    print("🌐 Acesse: http://localhost:5000")
    print("============================================================")
    print()
    print("Recursos disponíveis:")
    print("• Download direto do YouTube em FLAC 24kHz Mono")
    print("• Normalização de áudio com FFmpeg")
    print("• Segmentação natural baseada em pausas da fala (10s-1min)")
    print("• Filtro de qualidade MOS (threshold: 2.5)")
    print("• Diarização com pyannote.audio 3.1")
    print("• Detecção de sobreposição de vozes (threshold: 90%)")
    print("• Separação por locutor")
    print("• STT dual: Whisper + WAV2VEC2")
    print("• Validação STT com Levenshtein (80% threshold)")
    print("• Denoiser com DeepFilterNet3")
    print("• Todos os arquivos salvos localmente")
    print()
    print("Pressione Ctrl+C para parar o servidor")
    print("============================================================")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
