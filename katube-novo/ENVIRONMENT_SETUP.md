# 🔐 **Configuração de Variáveis de Ambiente**

## 🎯 **Objetivo**
Manter as chaves privadas (como token do Hugging Face) seguras e fora do controle de versão.

---

## 📋 **Passo a Passo**

### **1. Criar arquivo `.env`:**
```bash
# No diretório raiz do projeto, crie o arquivo .env
# (Ele já está no .gitignore e NÃO será commitado)

# Windows (PowerShell):
New-Item -Path ".env" -ItemType File

# Linux/Mac:
touch .env
```

### **2. Adicionar suas chaves no `.env`:**
```env
# ================================================
# HUGGING FACE AUTHENTICATION (OBRIGATÓRIO)
# ================================================
HUGGINGFACE_TOKEN=sua_chave_do_hugging_face_aqui

# ================================================
# CONFIGURAÇÕES OPCIONAIS
# ================================================
# Diretório personalizado (opcional)
AUDIOS_BAIXADOS_DIR=C:\seu\caminho\personalizado

# Configurações de áudio (opcional)
SAMPLE_RATE=24000
AUDIO_FORMAT=flac

# Configurações de segmentação (opcional)  
SEGMENT_MIN_DURATION=10.0
SEGMENT_MAX_DURATION=15.0
```

### **3. Obter Token do Hugging Face:**
1. 🌐 Acesse: https://huggingface.co/settings/tokens
2. 🔑 Clique em "New token"
3. 📝 Nome: `katube-projeto`
4. 🔓 Tipo: `Read`
5. 📋 Copie o token gerado

### **4. Aceitar Termos dos Modelos:**
- 🔗 https://huggingface.co/pyannote/speaker-diarization-3.1
- 🔗 https://huggingface.co/pyannote/segmentation-3.0  
- 🔗 https://huggingface.co/pyannote/embedding

---

## ✅ **Exemplo de `.env` Completo**
```env
# Sua chave do Hugging Face (OBRIGATÓRIO)
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Configurações opcionais
AUDIOS_BAIXADOS_DIR=C:\Users\SeuUsuario\Desktop\katube-novo\audios_baixados
SAMPLE_RATE=24000
FLASK_DEBUG=True
```

---

## 🚀 **Testando a Configuração**
```python
# Teste se está funcionando:
from src.config import Config

print("Token carregado:", "✅" if Config.HUGGINGFACE_TOKEN else "❌")
print("Diretório:", Config.AUDIOS_BAIXADOS_DIR)
```

---

## 🔒 **Segurança**

### ✅ **O que FAZER:**
- ✅ Criar `.env` localmente
- ✅ Adicionar suas chaves reais no `.env`
- ✅ Usar `env.example` como referência

### ❌ **O que NÃO fazer:**
- ❌ **NUNCA** faça commit do `.env`
- ❌ **NUNCA** compartilhe suas chaves
- ❌ **NUNCA** coloque chaves no código

---

## 🆘 **Solução de Problemas**

### **Erro: "HUGGINGFACE_TOKEN not found"**
```bash
# Verificar se o arquivo existe:
dir .env    # Windows
ls -la .env # Linux/Mac

# Verificar conteúdo (sem expor a chave):
type .env | findstr HUGGINGFACE_TOKEN    # Windows  
cat .env | grep HUGGINGFACE_TOKEN        # Linux/Mac
```

### **Erro: "401 Unauthorized"**
- ✅ Verificar se aceitou termos dos modelos
- ✅ Verificar se o token está correto
- ✅ Regenerar token se necessário

---

## 📦 **Para Distribuição**
Quando compartilhar o projeto, lembre de incluir:
- ✅ `env.example` (template)
- ✅ `ENVIRONMENT_SETUP.md` (este arquivo)
- ✅ Instruções no README principal
