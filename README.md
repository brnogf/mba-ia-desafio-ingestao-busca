# 🔍 Sistema de Ingestão e Busca Semântica de PDFs (RAG)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://python.langchain.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgVector-316192)](https://github.com/pgvector/pgvector)

Este projeto implementa uma arquitetura de **RAG (Retrieval-Augmented Generation)**. O sistema realiza a ingestão de documentos PDF, processa o texto, gera *embeddings* e armazena os dados de forma vetorial. A interface de busca via CLI permite recuperar os contextos mais relevantes e formular respostas baseadas **estritamente** no domínio do documento fornecido, mitigando *alucinações* do modelo de linguagem.

---

## 🏗️ Arquitetura da Solução

O fluxo da aplicação foi desenhado em duas pipelines principais:

1. **Pipeline de Ingestão (`src/ingest.py`):**
   - **Extração:** Leitura do arquivo `document.pdf`.
   - **Chunking:** O texto é dividido em janelas de 1000 caracteres com *overlap* de 150 caracteres (usando `RecursiveCharacterTextSplitter`). O overlap é crucial para evitar a perda de contexto semântico nas bordas dos parágrafos.
   - **Vetorização e Armazenamento:** Os chunks são convertidos em vetores de alta dimensionalidade via API (OpenAI/Gemini) e persistidos em um banco PostgreSQL utilizando a extensão `pgVector`.

2. **Pipeline de Busca e Geração (`src/chat.py` & `src/search.py`):**
   - **User Query:** Captura da pergunta do usuário via CLI.
   - **Retrieval:** A query é vetorizada e uma busca de similaridade (Cosine/L2) é feita no pgVector para recuperar os *Top K* (K=10) chunks mais relevantes.
   - **Augmented Generation:** O contexto recuperado é injetado no *System Prompt* com *guardrails* rígidos, instruindo a LLM a responder unicamente com base no contexto, ou retornar uma mensagem padrão de segurança caso a resposta não esteja no escopo.

---

## 🛠️ Decisões Técnicas e Premissas

- **Framework RAG:** Escolha do **LangChain** pela abstração robusta na orquestração de LLMs e integração nativa com o ecossistema de Vector Stores.
- **Armazenamento Vetorial:** Adoção do **PostgreSQL + pgVector** ao invés de bancos NoSQL puramente vetoriais (como Pinecone ou Chroma) por garantir persistência relacional transacional aliada à busca semântica em um único contêiner local, facilitando a portabilidade do ambiente.
- **Modelos Escolhidos:** Optou-se pela utilização das bibliotecas padrão sugeridas (Google Gemini ou OpenAI) visando o balanço ideal entre latência, custo e qualidade de embedding para textos.

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
> **Atenção:** Edite o arquivo `.env` inserindo sua respectiva chave na variável (ex: `OPENAI_API_KEY` ou a correspondente do Gemini).

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

---

## 🛑 Troubleshooting e Mapeamento de Exceções

Caso encontre problemas durante o setup ou uso, consulte os cenários abaixo:

- **Erro de Autenticação na API (HTTP 401 / 429 - Quota Exceeded):**
  - *Cenário:* A execução de `ingest.py` ou `chat.py` retorna falhas de autenticação com a LLM.
  - *Causa:* A API Key contida no `.env` foi revogada, expirou ou o limite de requisições gratuitas da conta foi atingido.
  - *Solução:* Gere uma nova chave de API no painel do seu provedor, atualize o arquivo `.env` e certifique-se de que a conta possui saldo/créditos ativos.

- **Conflito de Dimensionalidade Vetorial no pgVector (Dimension Mismatch):**
  - *Cenário:* Ao rodar `ingest.py`, o PostgreSQL lança uma exceção informando que as dimensões do vetor não coincidem.
  - *Causa:* O banco de dados cria a estrutura da tabela baseado na dimensão do primeiro modelo de *embedding* utilizado. Se você alterar a API para outro modelo (ex: de OpenAI para Gemini) no meio do desenvolvimento, os novos vetores terão tamanho diferente, gerando *crash* na inserção.
  - *Solução:* O estado do banco deve ser recriado. Derrube o volume Docker e suba novamente:
    ```bash
    docker compose down -v
    docker compose up -d
    python src/ingest.py
    ```

- **Falha de Conexão com Banco / Porta 5432 Ocupada:**
  - *Cenário:* O contêiner Docker do PostgreSQL não sobe ou a aplicação acusa falha de TCP.
  - *Causa:* Um serviço local (geralmente uma instalação nativa do Postgres) já está escutando na porta 5432.
  - *Solução:* Altere o *bind port* no `docker-compose.yml` (ex: `5433:5432`) e atualize a *Connection String* no código para apontar para a nova porta.