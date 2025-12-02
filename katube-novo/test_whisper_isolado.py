import sys
from pathlib import Path

# Adicionar src/ ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from stt_whisper import WhisperSTTTranscriber

print("=== TESTE WHISPER ISOLADO ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Inicializar Whisper
print("\n📦 Inicializando Whisper...")
whisper = WhisperSTTTranscriber(device="cuda")
print(f"✅ Whisper device: {whisper.device}")

# Testar com UM segmento
test_audio = Path("audios_baixados/output/EhzSC3LWez4_backup/stt_ready/speaker_SPEAKER_00/EhzSC3LWez4_segment_000_SPEAKER_00_1.43_24.41.flac")

if not test_audio.exists():
    print(f"❌ Arquivo não encontrado: {test_audio}")
    # Tentar encontrar outro
    import glob
    alts = glob.glob("audios_baixados/output/*/stt_ready/speaker_*/*.flac")
    if alts:
        test_audio = Path(alts[0])
        print(f"📁 Usando alternativo: {test_audio}")
    else:
        print("❌ Nenhum áudio disponível")
        exit(1)

print(f"\n🎤 Transcrevendo: {test_audio.name}")
print("⏱️ Iniciando...")

import time
start = time.time()

try:
    text = whisper.transcribe_audio(test_audio)
    elapsed = time.time() - start
    
    print(f"\n✅ SUCESSO!")
    print(f"⏱️ Tempo: {elapsed:.2f}s")
    print(f"📝 Transcrição ({len(text)} chars): {text[:100]}...")
    
except Exception as e:
    elapsed = time.time() - start
    print(f"\n❌ ERRO após {elapsed:.2f}s")
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()
