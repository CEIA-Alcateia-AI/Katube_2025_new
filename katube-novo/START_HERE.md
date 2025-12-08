
1. ✅ **Colar um link do YouTube** na interface web
2. ✅ **Processar automaticamente** por trás (download → segmentação → diarização → separação)  
3. ✅ **Acompanhar o progresso** em tempo real
4. ✅ **Baixar os resultados** organizados por locutor
5. ✅ **Arquivos salvos** em `C:\Users\Usuário\Desktop\katube-novo\audios_baixados`

## ⚡ INÍCIO IMEDIATO - 3 Passos

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar Token Hugging Face
```bash
export HUGGINGFACE_TOKEN="seu_token_aqui"
```
> 📝 **Como obter o token:**
> 1. Vá para https://huggingface.co/settings/tokens
> 2. Crie um token novo
> 3. Aceite os termos em https://huggingface.co/pyannote/speaker-diarization-3.1

### 3. Iniciar o Servidor
```
python app.py
```

**Abra no navegador:** http://localhost:5000

## 🎯 Como Usar a Interface

### Página Principal
1. **Cole a URL do YouTube** no campo
2. **Configure opções avançadas** (opcional):
   - Nome personalizado
   - Número de locutores esperados  
   - Duração dos segmentos
3. **Clique em "Processar Áudio"**

### Acompanhar Progresso
- ⏳ **Barra de progresso** visual
- 📊 **Steps em tempo real**: Download → Segmentação → Diarização → etc.
- 💬 **Mensagens descritivas** de cada etapa

### Página de Resultados
- 📈 **Estatísticas completas**: Locutores encontrados, arquivos criados
- 👥 **Detalhes por locutor**: Duração, número de arquivos
- 💾 **Downloads disponíveis**: 
  - JSON com todos os dados
  - ZIP por locutor (prontos para STT!)

## 📁 Estrutura dos Arquivos Gerados

```
audios_baixados/output/web_session_*/
├── downloads/                    # Etapa 1: Download
├── normalized/                   # Etapa 2: Normalização
├── segments/                     # Etapa 3: Segmentação
├── segments_aprovados/           # Etapa 4: MOS Aprovados
├── audio_descartado_mos/         # Etapa 4: MOS Rejeitados
├── diarization/                  # Etapa 5: Diarização
├── overlap/                      # Etapa 6: Overlap Detection
├── speakers/                     # Etapa 7: Separação
└── stt_results/                  # Etapas 8-10: STT, Validação, Denoiser
    ├── whisper/                  # STT Whisper
    ├── wav2vec2/                 # STT WAV2VEC2
    ├── validation_results/       # Validação
    ├── audios_validados_tts/     # Validados (≥80%)
    ├── audio_rejeitado_validacao/ # Rejeitados (<80%)
    └── audios_denoiser/          # Denoised
```