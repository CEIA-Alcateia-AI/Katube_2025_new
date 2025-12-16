# Visualizador de Espectrograma

Script para visualizar o espectrograma de arquivos de áudio e verificar se a qualidade foi preservada.

## Uso Básico

```bash
# Visualizar espectrograma (abre janela)
python visualize_spectrogram.py audio.flac

# Salvar espectrograma como imagem
python visualize_spectrogram.py audio.flac --save output.png

# Com tamanho de janela customizado
python visualize_spectrogram.py audio.flac --window-size 4096

# Exemplos práticos
python visualize_spectrogram.py audio\hxx49fdPQdI_Imposto_nos_dividendos_Veja_o_que_o_Haddad_Andrade_disse_a_respeito.flac
python visualize_spectrogram.py audio.flac --save spectrogram.png
```

## O Que o Script Mostra

1. **Informações do Áudio**:
   - Sample Rate
   - Canais
   - Duração
   - Nyquist Frequency
   - RMS e Peak

2. **Análise de Frequências**:
   - Frequência máxima com energia significativa
   - Se há conteúdo acima de 8kHz (indicando >16kHz)
   - Se há conteúdo acima de 12kHz (indicando >24kHz)
   - Diagnóstico automático

3. **Visualização**:
   - Espectrograma colorido
   - Linhas de referência (Nyquist, 8kHz, 12kHz)
   - Gráfico de magnitude média

## Interpretando os Resultados

### Sample Rate 24kHz - CORRETO 
- Espectrograma mostra frequências até ~12kHz
- Há conteúdo frequencial acima de 12kHz
- **Diagnóstico**: "Sample Rate é 24kHz - qualidade preservada"

### Sample Rate 16kHz - PROBLEMA 
- Espectrograma mostra frequências apenas até ~8kHz
- Não há conteúdo acima de 8kHz
- **Diagnóstico**: "Sample Rate é 16kHz - áudio pode ter sido reamostrado"

### Indicadores Visuais no Gráfico

- **Linha Vermelha (tracejada)**: Nyquist Frequency (metade do sample rate)
- **Linha Laranja (tracejada)**: Limite de 8kHz (indicando 16kHz)
- **Linha Amarela (tracejada)**: Limite de 12kHz (indicando 24kHz)


