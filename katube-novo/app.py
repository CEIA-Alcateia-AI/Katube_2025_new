#!/usr/bin/env python3
"""
Flask web interface for YouTube Audio Processing Pipeline
"""
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
        job.update("segmenting", 25, "Segmentando áudio...")
        
        # Segment
        segments = pipeline.segment_audio(audio_path, options.get('intelligent_segmentation', True))
        job.update("diarizing", 40, "Realizando diarização de locutores...")
        
        # Diarization
        diarization_results = pipeline.perform_diarization(segments, options.get('num_speakers'))
        job.update("detecting", 60, "Detectando sobreposições de voz...")
        
        # Overlap detection
        clean_segments, overlapping_segments = pipeline.detect_overlaps(segments)
        job.update("separating", 80, "Separando áudios por locutor...")
        
        # Speaker separation
        separation_results = pipeline.separate_speakers(diarization_results, options.get('enhance_audio', True))
        job.update("preparing", 95, "Preparando arquivos para STT...")
        
        # STT preparation
        stt_files = pipeline.prepare_for_stt(separation_results)
        
        # Complete results
        processing_time = time.time() - job.start_time.timestamp()
        
        results = {
            'session_name': session_name,
            'session_dir': str(session_dir),
            'url': url,
            'processing_time': processing_time,
            'downloaded_audio': str(audio_path),
            'num_segments': len(segments),
            'num_clean_segments': len(clean_segments),
            'num_overlapping_segments': len(overlapping_segments),
            'stt_ready_files': stt_files,
            'statistics': pipeline._generate_statistics(stt_files, separation_results)
        }
        
        # Save results to JSON
        results_file = session_dir / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json_results = pipeline._prepare_for_json(results)
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        job.complete(results)
        
    except Exception as e:
        logger.error(f"Background job {job_id} failed: {e}")
        job.fail(str(e))

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

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
        
        return jsonify({
            'job_id': job.job_id,
            'status': job.status,
            'results': job.results,
            'processing_time': (job.end_time - job.start_time).total_seconds()
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
    
    print("🚀 Starting YouTube Audio Processing Web Interface")
    print(f"📁 Output directory: {Config.OUTPUT_DIR}")
    print("🌐 Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
