# 🔍 Sistema de Ingestão e Busca Semântica de PDFs (RAG)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://python.langchain.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgVector-316192)](https://github.com/pgvector/pgvector)

Este projeto implementa uma arquitetura de **RAG (Retrieval-Augmented Generation)** de alta fidelidade para o desafio da MBA Full Cycle. O sistema realiza a ingestão de documentos PDF, processa o texto, gera *embeddings* e armazena os dados de forma vetorial. A interface de busca via CLI permite recuperar os contextos mais relevantes e formular respostas baseadas **estritamente** no domínio do documento fornecido, mitigando *alucinações* do modelo de linguagem.

---

## 🏗️ Arquitetura da Solução

O fluxo da aplicação foi desenhado em duas pipelines principais:

1. **Pipeline de Ingestão (`src/ingest.py`):**
   - **Extração:** Leitura do arquivo `document.pdf` (34 páginas).
   - **Chunking:** O texto é dividido em janelas de 1000 caracteres com *overlap* de 150 caracteres (usando `RecursiveCharacterTextSplitter`). O overlap é crucial para evitar a perda de contexto semântico nas bordas dos parágrafos.
   - **Idempotência & Hash:** Rastreamento por SHA-256 para evitar consumo desnecessário de cotas de API caso o documento já tenha sido indexado.
   - **Vetorização e Armazenamento:** Os chunks são convertidos em vetores de alta dimensionalidade via API (**OpenAI** ou **Google Gemini**) e persistidos em um banco PostgreSQL utilizando a extensão `pgVector`.

2. **Pipeline de Busca e Geração (`src/chat.py` & `src/search.py`):**
   - **User Query:** Captura da pergunta do usuário via CLI.
   - **Retrieval:** A query é vetorizada e uma busca de similaridade (Cosine/L2) é feita no pgVector para recuperar os *Top K* (K=10) chunks mais relevantes.
   - **Augmented Generation:** O contexto recuperado é injetado no *System Prompt* com *guardrails* rígidos, instruindo a LLM a responder unicamente com base no contexto, ou retornar uma mensagem padrão de segurança caso a resposta não esteja no escopo.

---

## 🔑 Configuração dos Provedores de IA (OpenAI vs. Google Gemini)

O projeto possui **suporte dual transparente** e detecta automaticamente qual provedor você deseja utilizar.

### Como configurar no arquivo `.env`:

Crie seu arquivo `.env` a partir do modelo:
```bash
cp .env.example .env
```

#### Opção A: Utilizando OpenAI (Padrão de Mercado)
Basta preencher sua chave da OpenAI no `.env`:
```ini
OPENAI_API_KEY=sk-proj-...sua-chave-aqui...
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
OPENAI_CHAT_MODEL='gpt-4o-mini'
```

#### Opção B: Utilizando Google Gemini
Basta preencher sua chave do Google Gemini no `.env`:
```ini
GOOGLE_API_KEY=AIzaSy...sua-chave-aqui...
GOOGLE_EMBEDDING_MODEL='models/gemini-embedding-001'
GOOGLE_CHAT_MODEL='gemini-flash-lite-latest'
```

#### Opção C: Ambas as Chaves Preenchidas
Caso você tenha inserido ambas as chaves no mesmo arquivo `.env`, você pode alternar entre elas definindo a variável `AI_PROVIDER`:
```ini
AI_PROVIDER=openai   # Para forçar o uso da OpenAI
# ou
AI_PROVIDER=gemini   # Para forçar o uso do Google Gemini
```
*(Se `AI_PROVIDER` não for definido e ambas as chaves estiverem presentes, o sistema prioriza OpenAI por padrão).*

### Como conferir qual chave e provedor estão ativos:
1. **Ao iniciar o assistente (`python src/chat.py`):** O banner inicial exibe o **Provedor Ativo**, o modelo LLM, o modelo de Embeddings e a **Chave Mascarada** (ex: `sk-p...1234` ou `AIza...9876`).
2. **Dentro do chat interativo:** Digite `/info` para abrir o painel completo de auditoria do sistema, detalhando provedor ativo, dimensões e infraestrutura.

---

## 🛠️ Decisões Técnicas e Premissas

- **Framework RAG:** Escolha do **LangChain** pela abstração robusta na orquestração de LLMs e integração nativa com o ecossistema de Vector Stores.
- **Armazenamento Vetorial:** Adoção do **PostgreSQL + pgVector** ao invés de bancos NoSQL puramente vetoriais (como Pinecone ou Chroma) por garantir persistência relacional transacional aliada à busca semântica em um único contêiner local, facilitando a portabilidade do ambiente.
- **Modelos Escolhidos:** Optou-se pela utilização das bibliotecas oficiais (`langchain-openai` e `langchain-google-genai`) visando o balanço ideal entre latência, custo e qualidade de embedding.
- **Proteção contra Incompatibilidade de Dimensões:** A extensão `pgvector` exige que todos os vetores comparados tenham a mesma dimensão (ex: OpenAI usa 1536 dims, Gemini usa 768 ou 3072 dims). O pipeline possui **auto-recuperação 100% transparente**: ao detectar qualquer alteração de API key ou modelo de embeddings, o sistema sincroniza, limpa dados antigos e reindexa o banco automaticamente em background, sem exigir comandos manuais do usuário.

---

## 🚀 Como Executar o Projeto (Developer Experience)

### 1. Pré-requisitos do Ambiente
- Python 3.8+
- Docker e Docker Compose
- Chave de API válida (OpenAI ou Google Gemini)

### 2. Setup Inicial

Clone o repositório e configure as variáveis de ambiente:
```bash
git clone https://github.com/brnogf/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca

# Crie e preencha o arquivo de configuração de ambiente
cp .env.example .env
```

Configure o ambiente virtual isolado (recomendado):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências da aplicação:
```bash
pip install -r requirements.txt
```

### 3. Execução

Suba a infraestrutura do banco de dados vetorial:
```bash
docker compose up -d
```

Realize a carga (ingestão) do documento para o banco de dados:
```bash
python src/ingest.py
```

Inicie a interface de comunicação (CLI):
```bash
python src/chat.py
```

Ou execute consultas diretas via linha de comando:
```bash
python src/chat.py -q "Qual é o faturamento da empresa Alfa Agronegócio Indústria?"
```

---

## 🛑 Troubleshooting e Mapeamento de Exceções

- **Troca de Provedor de Embeddings (Dimension Mismatch / Auto-Recovery):**
  - *Cenário:* Você indexou o banco inicialmente com um provedor (ex: OpenAI) e posteriormente alterou o `.env` para usar outro (ex: Gemini).
  - *Comportamento do Sistema:* O sistema detecta a diferença de dimensões vetoriais e executa a sincronização e reindexação automaticamente de forma transparente. No terminal, apenas avisa que uma mudança foi detectada e que o procedimento de atualização está sendo executado.
  - *Ação do Usuário:* Nenhuma intervenção manual necessária. O próprio sistema conclui o processo e responde a consulta normalmente.

- **Erro de Autenticação na API (HTTP 401 / 429 - Quota Exceeded):**
  - *Cenário:* A execução de `ingest.py` ou `chat.py` retorna falhas de autenticação com a LLM.
  - *Causa:* A API Key contida no `.env` foi revogada, expirou ou a cota gratuita do provedor foi atingida.
  - *Solução:* Atualize a respectiva chave (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`) no arquivo `.env`.

- **Falha de Conexão com o Docker:**
  - *Cenário:* Ao rodar `docker compose up -d`, o terminal exibe erro de conexão com daemon.
  - *Solução:* Certifique-se de que o Docker Desktop está aberto e em execução.