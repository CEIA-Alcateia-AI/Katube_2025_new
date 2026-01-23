import os
import librosa
import soundfile as sf
from pathlib import Path
import logging
import warnings

# Configuração de Log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suprimir avisos do Librosa/PySoundFile se necessário
warnings.filterwarnings("ignore")

def resample_directory(root_dir: str, target_sr: int = 16000, extensions: list = ['.wav', '.flac', '.ogg', '.mp3']):
    """
    Percorre pastas recursivamente, verifica o SR e converte para target_sr se necessário.
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        logger.error(f"❌ Diretório não encontrado: {root_dir}")
        return

    logger.info(f"🚀 Iniciando verificação de SR ({target_sr}Hz) em: {root_path}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Percorre recursivamente todas as subpastas
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                # 1. Maneira RÁPIDA de checar o SR sem carregar o áudio inteiro na RAM
                original_sr = librosa.get_samplerate(str(file_path))
                
                if original_sr != target_sr:
                    logger.info(f"🔄 Convertendo {file_path.name}: {original_sr}Hz -> {target_sr}Hz")
                    
                    # 2. Carrega já fazendo o resample (O librosa cuida da matemática)
                    y, _ = librosa.load(str(file_path), sr=target_sr, mono=True)
                    
                    # 3. Sobrescreve o arquivo usando soundfile
                    # Nota: O formato será mantido ou convertido para FLAC/WAV dependendo da extensão
                    sf.write(str(file_path), y, target_sr)
                    
                    processed_count += 1
                else:
                    # logger.info(f"✅ Já está em {target_sr}Hz: {file_path.name}")
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Erro ao processar {file_path.name}: {e}")
                error_count += 1

    logger.info("="*50)
    logger.info("RESUMO DA OPERAÇÃO")
    logger.info("="*50)
    logger.info(f"✅ Arquivos convertidos: {processed_count}")
    logger.info(f"⏭️  Arquivos já corretos: {skipped_count}")
    logger.info(f"❌ Falhas: {error_count}")
    logger.info("="*50)

if __name__ == "__main__":
    # COLOQUE O CAMINHO DA SUA PASTA AQUI
    # Exemplo Windows: r"C:\Igor\BIA\Alcateia\audios_brutos"
    # Exemplo Linux: "/home/ubuntu/experimentos_fred/audios"
    
    target_folder = input("Digite o caminho da pasta raiz para verificar: ").strip().replace('"', '')
    
    resample_directory(target_folder)