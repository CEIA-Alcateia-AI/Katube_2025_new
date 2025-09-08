# 🚀 COMO USAR O FRONTEND - INÍCIO RÁPIDO

## 📋 O Que Foi Criado

Criei um **frontend web completo** para sua pipeline de processamento de áudio do YouTube! Agora você pode:

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
```bash
python run_server.py
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
C:\Users\Usuário\Desktop\katube-novo\audios_baixados\
└── output\
    └── session_20240101_120000\       # Sessão com timestamp
        ├── downloads\                 # Áudio original (FLAC)
        ├── segments\                  # Segmentos de 10-15s
        ├── diarization\               # Arquivos .rttm
        ├── speakers\                  # Separado por locutor
        └── stt_ready\                 # 🎯 ARQUIVOS FINAIS PARA STT
            ├── speaker_SPEAKER_00\    
            └── speaker_SPEAKER_01\    
```

## 🧪 Testar o Sistema

### Teste Básico (Recomendado)
```bash
python test_web_interface.py
# Escolha opção "2" para teste rápido
```

### Teste Completo (com processamento real)
```bash  
python test_web_interface.py
# Escolha opção "1" - vai processar um vídeo de teste
```

## 🎨 Recursos da Interface

### ✨ Visual
- **Design moderno** com gradientes e animações
- **Totalmente responsivo** (funciona no celular)
- **Progresso em tempo real** com steps visuais
- **Notificações e feedbacks** claros

### ⚡ Funcional
- **Validação de URLs** do YouTube
- **Configurações avançadas** dobráveis
- **Auto-save** dos formulários
- **Downloads organizados** por locutor
- **Cleanup automático** dos dados

### 🔧 Técnico
- **Threading** para não travar a interface
- **API REST** completa
- **Tratamento de erros** robusto
- **Logs detalhados** para debug

## 🛠️ Comandos Úteis

### Iniciar Servidor
```bash
# Método principal
python run_server.py

# Com debug
FLASK_DEBUG=1 python app.py

# Com configurações
FLASK_ENV=development python app.py
```

### Verificar Status
```bash
# Ver jobs ativos
curl http://localhost:5000/jobs

# Verificar servidor
curl http://localhost:5000
```

### Logs
```bash
# Ver logs em tempo real
tail -f pipeline.log
```

## 🎯 Exemplo de Uso Completo

1. **Abra** http://localhost:5000
2. **Cole** uma URL do YouTube (ex: podcast, entrevista, aula)
3. **Configure** (opcional):
   - Nome: "minha_entrevista" 
   - Locutores: 2
   - Segmentos: 12-18s
4. **Processe** e aguarde 5-10 minutos
5. **Baixe** os arquivos por locutor
6. **Use** os arquivos em `stt_ready/` para transcrição!

## 🚨 Solução de Problemas

### Porta em Uso
```bash
# Encontrar processo na porta 5000
netstat -ano | findstr :5000
# Parar processo
taskkill /PID [número] /F
```

### Token Inválido
```bash
# Verificar token
echo $HUGGINGFACE_TOKEN
# Ou editar diretamente em src/config.py linha 12
```

### Pasta Não Encontrada
```bash
# Criar manualmente se necessário
mkdir "C:\Users\Usuário\Desktop\katube-novo\audios_baixados"
```

## 📞 Endpoints da API

Se quiser usar programaticamente:

```python
import requests

# Iniciar processamento
response = requests.post('http://localhost:5000/process', json={
    'url': 'https://youtube.com/watch?v=...',
    'filename': 'meu_audio'
})
job_id = response.json()['job_id']

# Verificar status
status = requests.get(f'http://localhost:5000/status/{job_id}').json()

# Obter resultados  
results = requests.get(f'http://localhost:5000/result/{job_id}').json()
```

## 🎉 Pronto Para Usar!

Agora você tem um **frontend web completo** que:
- ✅ Recebe links do YouTube
- ✅ Processa automaticamente em background  
- ✅ Salva em `C:\Users\Usuário\Desktop\katube-novo\audios_baixados`
- ✅ Interface bonita e funcional
- ✅ Downloads organizados para STT

**Execute `python run_server.py` e divirta-se!** 🚀
