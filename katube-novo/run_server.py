#!/usr/bin/env python3
"""
Quick server launcher for development
"""
import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables
os.environ.setdefault('FLASK_ENV', 'development')
os.environ.setdefault('FLASK_DEBUG', '1')

# Import and run the Flask app
if __name__ == '__main__':
    from app import app
    from src.config import Config
    
    # Create necessary directories
    Config.create_directories()
    Config.AUDIOS_BAIXADOS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 YouTube Audio Processing Pipeline - Web Interface")
    print("=" * 60)
    print(f"📁 Audios serão salvos em: {Config.AUDIOS_BAIXADOS_DIR}")
    print(f"📂 Output directory: {Config.OUTPUT_DIR}")
    print("🌐 Acesse: http://localhost:5000")
    print("=" * 60)
    print()
    print("Recursos disponíveis:")
    print("• Download direto do YouTube em FLAC")
    print("• Segmentação inteligente (10-15s)")
    print("• Diarização com pyannote.audio 3.1")
    print("• Detecção de sobreposição de vozes")
    print("• Separação por locutor")
    print("• Arquivos prontos para STT")
    print()
    print("Pressione Ctrl+C para parar o servidor")
    print("=" * 60)
    
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro ao iniciar servidor: {e}")
        sys.exit(1)
