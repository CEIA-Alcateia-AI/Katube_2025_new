import shutil
import logging
from pathlib import Path
import sys

# Configuração de Log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def organize_single_directory(target_path: Path, extensions: list):
    """
    Função auxiliar: Organiza arquivos dentro de UMA pasta específica.
    Pega 'audio.ogg' e move para 'audio/audio.ogg'.
    """
    moved_count = 0
    
    # Itera sobre os arquivos da pasta
    for file_path in target_path.iterdir():
        # Verifica se é arquivo, se tem a extensão, e ignora scripts python
        if file_path.is_file() and file_path.suffix.lower() in extensions and not file_path.name.endswith(".py"):
            try:
                # 1. Define o nome da nova pasta (baseado no nome do arquivo)
                new_folder_name = file_path.stem
                new_folder_path = target_path / new_folder_name
                
                # 2. Cria a pasta
                new_folder_path.mkdir(exist_ok=True)
                
                # 3. Define destino
                destination_path = new_folder_path / file_path.name
                
                # 4. Move
                shutil.move(str(file_path), str(destination_path))
                
                logger.info(f"         ∟ ✅ {file_path.name} -> {new_folder_name}/")
                moved_count += 1
                
            except Exception as e:
                logger.error(f"         ∟ ❌ Erro em {file_path.name}: {e}")
    
    return moved_count

def process_grandparent_directory(root_dir_str: str, extensions: list = ['.ogg', '.flac', '.wav', '.mp3']):
    """
    Função Principal: Itera com profundidade de 2 níveis.
    Raiz -> Pasta Nível 1 -> Pasta Nível 2 -> [Organiza Arquivos Aqui]
    """
    root_path = Path(root_dir_str)
    
    if not root_path.exists():
        logger.error(f"❌ Diretório raiz não encontrado: {root_dir_str}")
        return

    logger.info(f"🚀 Iniciando organização PROFUNDA em: {root_path.name}")
    logger.info(f"{'='*60}")

    total_moved = 0
    folders_processed = 0

    # NÍVEL 1: Itera sobre as pastas dentro da raiz
    for level1_folder in root_path.iterdir():
        if level1_folder.is_dir() and not level1_folder.name.startswith('.'):
            logger.info(f"📂 Processando Grupo: {level1_folder.name}")
            
            # NÍVEL 2: Itera sobre as pastas dentro do nível 1
            has_subfolders = False
            for level2_folder in level1_folder.iterdir():
                if level2_folder.is_dir() and not level2_folder.name.startswith('.'):
                    has_subfolders = True
                    # logger.info(f"   📂 Verificando pasta final: {level2_folder.name}")
                    
                    # AÇÃO: Organiza os arquivos aqui dentro
                    count = organize_single_directory(level2_folder, extensions)
                    
                    if count > 0:
                        total_moved += count
                        folders_processed += 1
                        logger.info(f"   ✅ Organizado: {level2_folder.name} ({count} arquivos)")
            
            if not has_subfolders:
                logger.warning(f"   ⚠️ Nenhuma subpasta encontrada em {level1_folder.name}")

    logger.info(f"{'='*60}")
    logger.info(f"✨ Concluído! {total_moved} arquivos organizados em {folders_processed} pastas finais.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        print("Uso: python organize_deep_batch.py <caminho_da_pasta_raiz>")
        target_dir = input("Digite o caminho da pasta RAIZ (Avô): ").strip()

    # Limpeza de aspas
    target_dir = target_dir.replace('"', '').replace("'", "")
    
    process_grandparent_directory(target_dir)