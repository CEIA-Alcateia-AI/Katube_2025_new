# 🚨 Solução para Erro do Pyannote

## ❌ Problema
```
Could not download 'pyannote/speaker-diarization-3.1' pipeline.
'NoneType' object has no attribute 'to'
```

## ✅ Solução - Aceitar Termos dos Modelos

### 1. Acesse TODOS estes links e aceite os termos:

🔗 **Links obrigatórios:**
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0  
- https://huggingface.co/pyannote/embedding

### 2. Para cada link:
1. **Faça login** com sua conta Hugging Face
2. **Clique em "Agree and access repository"**
3. **Aceite todos os termos**

### 3. Após aceitar todos os termos:
```bash
# Pare o servidor (Ctrl+C)
# Reinicie:
python run_server.py
```

## 📝 Por que isso acontece?

- Os modelos pyannote são **"gated"** (restritos)
- Mesmo com token válido, precisa **aceitar termos**
- É uma exigência de licença dos pesquisadores

## 🧪 Como testar se funcionou:

Após aceitar os termos, você verá no log:
```
✅ Loaded pyannote/speaker-diarization-3.1 pipeline
```

Em vez de:
```
❌ Failed to load diarization pipeline
```

## 🔧 Status dos Termos

Marque conforme for aceitando:

- [ ] pyannote/speaker-diarization-3.1 ✅ OBRIGATÓRIO
- [ ] pyannote/segmentation-3.0 ✅ OBRIGATÓRIO  
- [ ] pyannote/embedding ✅ OBRIGATÓRIO

## ⚡ Depois que funcionar:

A pipeline completa estará disponível:
- ✅ Download do YouTube  
- ✅ Segmentação inteligente
- ✅ **Diarização funcionando** 🎯
- ✅ Detecção de sobreposição
- ✅ Separação por locutor
- ✅ Arquivos STT prontos
