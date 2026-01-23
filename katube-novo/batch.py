import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json
import sys
import argparse
from typing import List, Dict, Tuple

class BatchProcessor:
    """
    Processador em lote para pipeline de audio.
    Descobre recursivamente pastas contendo arquivos .ogg e processa sequencialmente.
    """
    
    def __init__(self, 
                 input_dir: Path = None,
                 log_dir: Path = None,
                 historico_dir: Path = None,
                 dry_run: bool = False):
        """
        Inicializa o processador em lote.
        """
        # Diretorios
        self.input_dir = Path(input_dir) if input_dir else Path("audios")
        self.log_dir = Path(log_dir) if log_dir else Path("dataset/log")
        self.historico_dir = Path(historico_dir) if historico_dir else Path("dataset/historico_dataset")
        
        # Modo dry-run
        self.dry_run = dry_run
        
        # Criar diretorios se nao existirem
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.historico_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        self._setup_logging()
        
        # Estatisticas
        self.stats = {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
    
    def _setup_logging(self):
        """Configura sistema de logging."""
        log_file = self.log_dir / "batch_processing.log"
        
        # Formato do log
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*70)
        self.logger.info("BATCH PROCESSOR INICIADO (SPOTIFY RECURSIVE MODE)")
        self.logger.info(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Input: {self.input_dir.absolute()}")
        self.logger.info(f"Log: {log_file.absolute()}")
        self.logger.info(f"Historico: {self.historico_dir.absolute()}")
        if self.dry_run:
            self.logger.info("MODO DRY-RUN: Nenhum processamento sera executado")
        self.logger.info("="*70)
    
    def discover_audio_folders(self) -> List[Path]:
        """
        Descobre pastas contendo arquivos de audio de forma RECURSIVA.
        Ideal para estrutura: Raiz -> Show -> Pasta_Audio -> Audio.ogg
        
        Returns:
            Lista de caminhos das pastas que contem audios
        """
        if not self.input_dir.exists():
            self.logger.error(f"Diretorio de entrada nao existe: {self.input_dir}")
            return []
        
        self.logger.info(f"🔍 Buscando arquivos .ogg recursivamente em: {self.input_dir}")
        
        # Busca recursiva (rglob) por qualquer arquivo .ogg
        # Isso garante que encontraremos o audio não importa a profundidade da pasta
        ogg_files = list(self.input_dir.rglob("*.ogg"))
        
        # Extrai apenas os diretórios pais (onde o arquivo está) e remove duplicatas
        # Ex: se tiver /show/pasta1/audio.ogg, ele pega /show/pasta1
        audio_folders = list(set([f.parent for f in ogg_files]))
        
        # Ordenar para processamento consistente
        audio_folders.sort()
        
        self.logger.info(f"✅ Descobertas {len(audio_folders)} pastas prontas para processamento")
        if len(audio_folders) > 0:
            self.logger.info(f"   Exemplo: {audio_folders[0].name}")
        
        return audio_folders
    
    def is_already_processed(self, folder_id: str) -> bool:
        """Verifica se uma pasta ja foi processada checando historico."""
        # Verifica json final no historico
        json_file = self.historico_dir / f"{folder_id}.json"
        
        # Opcional: Verifica tambem se ja existe CSV no dataset (se quiser ser mais rigoroso)
        # Mas o JSON no historico é o sinal padrao de sucesso da sua pipeline
        
        if json_file.exists():
            self.logger.info(f"  [SKIP] {folder_id} - Ja processado (encontrado {json_file.name})")
            return True
        
        return False
    
    def process_single_audio(self, folder_path: Path) -> Tuple[bool, str]:
        """Processa um unico audio chamando run_pipeline.py."""
        folder_id = folder_path.name
        
        self.logger.info(f"▶️ Processando: {folder_id}")
        
        if self.dry_run:
            self.logger.info(f"  [DRY-RUN] Simulando processamento de {folder_id}")
            return True, "Dry-run - nao processado"
        
        try:
            # Verifica se o script da pipeline existe
            pipeline_script = Path("run_pipeline.py")
            if not pipeline_script.exists():
                return False, "Script run_pipeline.py nao encontrado na raiz"

            # Construir comando
            cmd = [
                sys.executable,  # Garante usar o mesmo python do ambiente virtual atual
                "run_pipeline.py",
                str(folder_path),
                "--session-name",
                folder_id
            ]
            
            # Criar arquivo de log individual para este audio na pasta de logs
            pipeline_log_file = self.log_dir / f"{folder_id}_pipeline.log"
            
            self.logger.info(f"  Executando pipeline...")
            
            # Executar run_pipeline.py redirecionando output para arquivo
            with open(pipeline_log_file, 'w', encoding='utf-8') as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=Path.cwd() 
                )
            
            # Verificar resultado
            if result.returncode == 0:
                self.logger.info(f"  ✅ [OK] Sucesso! Log salvo em: {pipeline_log_file.name}")
                return True, "Sucesso"
            else:
                error_msg = f"Exit code {result.returncode}"
                self.logger.error(f"  ❌ [ERRO] Falha no processamento. Verifique: {pipeline_log_file.name}")
                return False, error_msg
        
        except Exception as e:
            error_msg = f"Excecao: {str(e)}"
            self.logger.error(f"  ❌ [ERRO CRITICO] {folder_id}: {error_msg}")
            return False, error_msg
    
    def process_all(self) -> Dict:
        """Processa todos os audios encontrados."""
        audio_folders = self.discover_audio_folders()
        
        if not audio_folders:
            self.logger.warning("⚠️ Nenhuma pasta com audio encontrada. Verifique o caminho de entrada.")
            return self.stats
        
        self.stats['total'] = len(audio_folders)
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"INICIANDO PROCESSAMENTO DE {self.stats['total']} AUDIOS")
        self.logger.info("="*70)
        
        for i, folder_path in enumerate(audio_folders, 1):
            folder_id = folder_path.name
            
            self.logger.info("")
            self.logger.info(f"📦 [{i}/{self.stats['total']}] {folder_id}")
            self.logger.info("-" * 30)
            
            # Verificar se ja foi processado
            if self.is_already_processed(folder_id):
                self.stats['skipped'] += 1
                continue
            
            # Processar
            success, message = self.process_single_audio(folder_path)
            
            if success:
                self.stats['processed'] += 1
            else:
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'folder': folder_id,
                    'error': message
                })
        
        self._print_summary()
        return self.stats
    
    def _print_summary(self):
        """Imprime resumo do processamento."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("RESUMO DO PROCESSAMENTO EM BATCH")
        self.logger.info("="*70)
        self.logger.info(f"Total encontrado:   {self.stats['total']}")
        self.logger.info(f"✅ Sucesso:         {self.stats['processed']}")
        self.logger.info(f"⏭️  Pulados (Feitos): {self.stats['skipped']}")
        self.logger.info(f"❌ Falharam:        {self.stats['failed']}")
        
        if self.stats['errors']:
            self.logger.info("")
            self.logger.info("RELATORIO DE ERROS:")
            for error in self.stats['errors']:
                self.logger.info(f"  - {error['folder']}: {error['error']}")
        
        self.logger.info("="*70)

def main():
    parser = argparse.ArgumentParser(description='Spotify Batch Processor')
    
    parser.add_argument(
        'input_dir', 
        help='Caminho da pasta RAIZ contendo os downloads (ex: /path/to/downloaded_segments_2/0)'
    )
    
    parser.add_argument('--dry-run', action='store_true', help='Simula sem processar')
    
    args = parser.parse_args()
    
    # Ajuste de caminhos padrão do seu projeto
    log_dir = Path("dataset/log")
    historico_dir = Path("dataset/historico_dataset")
    
    processor = BatchProcessor(
        input_dir=args.input_dir,
        log_dir=log_dir,
        historico_dir=historico_dir,
        dry_run=args.dry_run
    )
    
    stats = processor.process_all()
    
    if stats['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()