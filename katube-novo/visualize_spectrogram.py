"""
Script para visualizar espectrograma de arquivo de áudio.

Este script permite visualizar o espectrograma de um áudio para verificar
se a qualidade foi mantida (24kHz) ou degradada (16kHz).

Uso:
    python visualize_spectrogram.py <caminho_do_audio> [--save <output.png>] [--window-size N]
    
Exemplo:
    python visualize_spectrogram.py audio.flac
    python visualize_spectrogram.py audio.flac --save spectrogram.png
    python visualize_spectrogram.py audio.flac --window-size 2048
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import librosa
import argparse

# Configurar backend do matplotlib
# Backend não-interativo (Agg) - permite salvar sem abrir janela
# Será mudado para interativo apenas se necessário (sem --save)
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo por padrão

def load_audio(audio_path: Path):
    """Carrega áudio usando soundfile (preserva sample rate original)."""
    try:
        audio, sr = sf.read(str(audio_path))
        return audio, sr
    except Exception as e:
        print(f"Erro ao carregar com soundfile: {e}")
        print("Tentando com librosa...")
        audio, sr = librosa.load(str(audio_path), sr=None)
        return audio, sr


def get_audio_info(audio_path: Path, audio, sr):
    """Obtém informações sobre o áudio."""
    duration = len(audio) / sr
    
    # Converter para mono se necessário
    if len(audio.shape) > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    
    # Calcular estatísticas
    max_freq = sr / 2  # Nyquist frequency
    rms = np.sqrt(np.mean(audio_mono**2))
    peak = np.max(np.abs(audio_mono))
    
    return {
        'sample_rate': sr,
        'duration': duration,
        'channels': audio.shape[1] if len(audio.shape) > 1 else 1,
        'nyquist_frequency': max_freq,
        'rms': rms,
        'peak': peak,
        'audio_mono': audio_mono
    }


def compute_spectrogram(audio, sr, n_fft=2048, hop_length=512):
    """Calcula espectrograma usando STFT."""
    # Converter para mono se necessário
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Calcular STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    # Converter para magnitude em dB
    magnitude = np.abs(stft)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Frequências correspondentes
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Tempos correspondentes
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
    
    return magnitude_db, frequencies, times


def plot_spectrogram(magnitude_db, frequencies, times, info, audio_path, save_path=None):
    """Plota o espectrograma."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Espectrograma
    im = ax1.pcolormesh(times, frequencies, magnitude_db, shading='gouraud', cmap='viridis')
    ax1.set_xlabel('Tempo (segundos)', fontsize=12)
    ax1.set_ylabel('Frequência (Hz)', fontsize=12)
    ax1.set_title(f'Espectrograma: {audio_path.name}\nSample Rate: {info["sample_rate"]} Hz | '
                  f'Duração: {info["duration"]:.2f}s | Nyquist: {info["nyquist_frequency"]:.0f} Hz',
                  fontsize=14, fontweight='bold')
    ax1.set_ylim(0, min(15000, info["nyquist_frequency"] + 1000))  # Limitar a 15kHz para visualização
    
    # Adicionar linha indicando Nyquist frequency
    ax1.axhline(y=info["nyquist_frequency"], color='red', linestyle='--', linewidth=2, 
                label=f'Nyquist Frequency ({info["nyquist_frequency"]:.0f} Hz)')
    
    # Linha indicando 8kHz (limite de 16kHz)
    if info["nyquist_frequency"] > 8000:
        ax1.axhline(y=8000, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                    label='Limite 16kHz (8kHz Nyquist)')
    
    # Linha indicando 12kHz (limite de 24kHz)
    if info["nyquist_frequency"] > 12000:
        ax1.axhline(y=12000, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7,
                    label='Limite 24kHz (12kHz Nyquist)')
    
    ax1.legend(loc='upper right')
    plt.colorbar(im, ax=ax1, label='Magnitude (dB)')
    
    # Gráfico de frequência média ao longo do tempo
    mean_freq = np.mean(magnitude_db, axis=0)
    ax2.plot(times, mean_freq, 'b-', linewidth=1)
    ax2.set_xlabel('Tempo (segundos)', fontsize=12)
    ax2.set_ylabel('Magnitude média (dB)', fontsize=12)
    ax2.set_title('Magnitude Espectral Média ao Longo do Tempo', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Salvar e fechar figura (não abre janela)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        # Mostrar janela apenas se não estiver salvando
        plt.show()


def analyze_frequency_content(magnitude_db, frequencies, info):
    """Analisa o conteúdo de frequência do áudio."""
    print("\n" + "="*60)
    print("📊 ANÁLISE DE FREQUÊNCIAS")
    print("="*60)
    
    # Encontrar frequência máxima com energia significativa (acima de -60dB)
    threshold_db = -60
    significant_energy = magnitude_db > threshold_db
    
    if np.any(significant_energy):
        max_freq_idx = np.max(np.where(significant_energy)[0])
        max_freq = frequencies[max_freq_idx]
        print(f"Frequência máxima com energia significativa: {max_freq:.0f} Hz")
        
        # Verificar se há conteúdo acima de 8kHz (indicando >16kHz)
        has_8khz_content = np.any(magnitude_db[frequencies > 8000] > threshold_db)
        print(f"Conteúdo acima de 8kHz: {'✅ Sim' if has_8khz_content else '❌ Não'}")
        
        # Verificar se há conteúdo acima de 12kHz (indicando >24kHz)
        has_12khz_content = np.any(magnitude_db[frequencies > 12000] > threshold_db)
        print(f"Conteúdo acima de 12kHz: {'✅ Sim' if has_12khz_content else '❌ Não'}")
        
        # Diagnóstico
        print("\n🔍 DIAGNÓSTICO:")
        if info["sample_rate"] == 16000:
            print("   ⚠️  Sample Rate é 16kHz - esperado para modelos ML")
            print("   ⚠️  Áudio pode ter sido reamostrado")
        elif info["sample_rate"] == 24000:
            print("   ✅ Sample Rate é 24kHz - qualidade preservada")
            if has_12khz_content:
                print("   ✅ Há conteúdo frequencial acima de 12kHz - qualidade mantida")
            else:
                print("   ⚠️  Não há conteúdo frequencial acima de 12kHz")
                print("   ⚠️  Pode indicar que o áudio foi reamostrado para 16kHz")
        else:
            print(f"   ℹ️  Sample Rate é {info['sample_rate']}Hz")
            if max_freq < info["nyquist_frequency"] * 0.8:
                print("   ⚠️  Frequência máxima está abaixo do esperado para este sample rate")
    else:
        print("❌ Não foi possível detectar energia significativa no áudio")
    
    print("="*60)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Visualiza espectrograma de arquivo de áudio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python visualize_spectrogram.py audio.flac
  python visualize_spectrogram.py audio.flac --save output.png
  python visualize_spectrogram.py audio.flac --window-size 4096
        """
    )
    
    parser.add_argument('audio_path', type=str, help='Caminho para o arquivo de áudio')
    parser.add_argument('--save', type=str, default=None, help='Salvar espectrograma como imagem (ex: output.png)')
    parser.add_argument('--window-size', type=int, default=2048, help='Tamanho da janela FFT (padrão: 2048)')
    parser.add_argument('--hop-length', type=int, default=None, help='Hop length para STFT (padrão: window_size // 4)')
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_path)
    
    # Validar arquivo
    if not audio_path.exists():
        print(f"❌ Arquivo não encontrado: {audio_path}")
        sys.exit(1)
    
    # Carregar áudio
    print(f"\n🔍 Carregando áudio: {audio_path.name}")
    try:
        audio, sr = load_audio(audio_path)
    except Exception as e:
        print(f"❌ Erro ao carregar áudio: {e}")
        sys.exit(1)
    
    # Obter informações
    info = get_audio_info(audio_path, audio, sr)
    
    # Imprimir informações
    print("\n" + "="*60)
    print("📊 INFORMAÇÕES DO ÁUDIO")
    print("="*60)
    print(f"Arquivo: {audio_path.name}")
    print(f"Caminho: {audio_path.absolute()}")
    print(f"Sample Rate: {info['sample_rate']} Hz")
    print(f"Canais: {info['channels']}")
    print(f"Duração: {info['duration']:.2f} segundos")
    print(f"Nyquist Frequency: {info['nyquist_frequency']:.0f} Hz")
    print(f"RMS: {info['rms']:.4f}")
    print(f"Peak: {info['peak']:.4f}")
    print("="*60)
    
    # Calcular espectrograma
    hop_length = args.hop_length or (args.window_size // 4)
    print(f"\n📈 Calculando espectrograma (window={args.window_size}, hop={hop_length})...")
    
    try:
        magnitude_db, frequencies, times = compute_spectrogram(
            info['audio_mono'], sr, n_fft=args.window_size, hop_length=hop_length
        )
    except Exception as e:
        print(f"❌ Erro ao calcular espectrograma: {e}")
        sys.exit(1)
    
    # Análise de frequências
    analyze_frequency_content(magnitude_db, frequencies, info)
    
    # Plotar
    if args.save:
        print(f"\n🎨 Gerando e salvando espectrograma (sem abrir janela)...")
        try:
            plot_spectrogram(magnitude_db, frequencies, times, info, audio_path, args.save)
            print(f"✅ Espectrograma salvo em: {Path(args.save).absolute()}")
        except Exception as e:
            print(f"❌ Erro ao salvar espectrograma: {e}")
            sys.exit(1)
    else:
        print(f"\n🎨 Gerando visualização...")
        # Tentar mudar para backend interativo para mostrar janela
        try:
            matplotlib.use('TkAgg')
            # Recarregar pyplot após mudar backend
            import importlib
            importlib.reload(plt)
            plot_spectrogram(magnitude_db, frequencies, times, info, audio_path, None)
            print("\n✅ Visualização exibida. Feche a janela para finalizar.")
        except:
            try:
                matplotlib.use('Qt5Agg')
                import importlib
                importlib.reload(plt)
                plot_spectrogram(magnitude_db, frequencies, times, info, audio_path, None)
                print("\n✅ Visualização exibida. Feche a janela para finalizar.")
            except Exception as e:
                print("⚠️  Não foi possível abrir janela interativa.")
                print("💡 Use --save <arquivo.png> para salvar como imagem:")
                print(f"   python visualize_spectrogram.py {args.audio_path} --save output.png")
                sys.exit(1)


if __name__ == "__main__":
    main()

