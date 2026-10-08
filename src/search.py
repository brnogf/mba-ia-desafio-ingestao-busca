import os
import sys
import re
from typing import Tuple, List, Optional, Union, Any
from dotenv import load_dotenv

load_dotenv()

# Configuração e Detecção de Provedor de IA (OpenAI vs. Google Gemini)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
AI_PROVIDER = os.getenv("AI_PROVIDER", "").strip().lower()

if not AI_PROVIDER:
    if OPENAI_API_KEY and not GOOGLE_API_KEY:
        AI_PROVIDER = "openai"
    elif GOOGLE_API_KEY and not OPENAI_API_KEY:
        AI_PROVIDER = "gemini"
    elif OPENAI_API_KEY and GOOGLE_API_KEY:
        AI_PROVIDER = "openai"
    else:
        AI_PROVIDER = "gemini"

if AI_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("⚠️ ERRO: Provedor OpenAI selecionado, mas OPENAI_API_KEY não foi configurada no .env!")
elif AI_PROVIDER == "gemini" and not GOOGLE_API_KEY:
    raise ValueError("⚠️ ERRO: Provedor Gemini selecionado, mas GOOGLE_API_KEY não foi configurada no .env!")

# Modelos configurados
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"

GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL") or "models/gemini-embedding-001"
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL") or "gemini-flash-lite-latest"

ACTIVE_EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL if AI_PROVIDER == "openai" else GOOGLE_EMBEDDING_MODEL
ACTIVE_CHAT_MODEL = OPENAI_CHAT_MODEL if AI_PROVIDER == "openai" else GOOGLE_CHAT_MODEL

# Mascaramento seguro de chaves para auditoria e interface
def get_masked_key() -> str:
    key = OPENAI_API_KEY if AI_PROVIDER == "openai" else GOOGLE_API_KEY
    if not key:
        return "NÃO CONFIGURADA"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"

# Parâmetros de Banco de Dados e RAG
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documentos_fullcycle")
TOP_K = int(os.getenv("TOP_K", "10"))

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# Template exigido estritamente pelo desafio Full Cycle
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

FALLBACK_RESPONSE = "Não tenho informações necessárias para responder sua pergunta."

# Singletons em memória para evitar recriação de conexões a cada pergunta
_CACHED_VECTOR_STORE: Optional[PGVector] = None
_CACHED_LLM: Optional[Union[ChatGoogleGenerativeAI, ChatOpenAI]] = None


def get_embeddings():
    """Instancia o modelo de embeddings de acordo com o provedor ativo."""
    if AI_PROVIDER == "openai":
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )
    else:
        return GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )


def get_vector_store() -> PGVector:
    """Inicializa uma única vez e reutiliza o pool de conexões do PostgreSQL."""
    global _CACHED_VECTOR_STORE
    if _CACHED_VECTOR_STORE is None:
        embeddings = get_embeddings()
        _CACHED_VECTOR_STORE = PGVector(
            embeddings=embeddings,
            collection_name=PG_VECTOR_COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
    return _CACHED_VECTOR_STORE


def get_llm() -> Union[ChatGoogleGenerativeAI, ChatOpenAI]:
    """Inicializa uma única vez a LLM com temperatura zero de acordo com o provedor."""
    global _CACHED_LLM
    if _CACHED_LLM is None:
        if AI_PROVIDER == "openai":
            _CACHED_LLM = ChatOpenAI(
                model=OPENAI_CHAT_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0.0,
            )
        else:
            _CACHED_LLM = ChatGoogleGenerativeAI(
                model=GOOGLE_CHAT_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.0,
                transport="rest",
            )
    return _CACHED_LLM


def search_context_and_pages(query: str, vector_store: PGVector) -> Tuple[str, List[int], float]:
    """
    Recupera os k=10 chunks mais relevantes e extrai as páginas citadas.
    Trata de forma elegante eventual incompatibilidade de dimensões vetoriais.
    """
    try:
        results = vector_store.similarity_search_with_score(query, k=TOP_K)
    except Exception as e:
        err_str = str(e).lower()
        if "different vector dimensions" in err_str or "dataerror" in err_str:
            raise RuntimeError(
                f"\n⚠️ [INCOMPATIBILIDADE DE VETORES DETECTADA]\n"
                f"O banco de dados foi populado com outro modelo de embedding diferente do ativo ({AI_PROVIDER.upper()}: {ACTIVE_EMBEDDING_MODEL}).\n"
                f"Para sincronizar o banco com o novo modelo, execute no terminal:\n"
                f"  python src/ingest.py --force\n"
            ) from e
        raise e
    
    if not results:
        return "", [], 1.0

    contexto = "\n\n".join([doc.page_content for doc, _score in results])
    paginas = sorted(list({doc.metadata.get("page", 0) + 1 for doc, _score in results if "page" in doc.metadata}))
    melhor_score = min([score for _doc, score in results])
    return contexto, paginas, melhor_score


def locate_exact_pages(query: str, answer: str, results: List[Tuple[Any, float]]) -> List[int]:
    """
    Identifica com precisão cirúrgica a página do documento onde a informação
    que respondeu à pergunta realmente está localizada.
    """
    if not results or FALLBACK_RESPONSE in answer:
        return []

    numbers = re.findall(r"[\d\.,]+", answer)
    sig_numbers = [n.strip(".,") for n in numbers if len(n.strip(".,")) >= 2]

    tokens = re.findall(r"\b[A-Za-z0-9_Á-ú]+\b", query) + re.findall(r"\b[A-Za-z0-9_Á-ú]+\b", answer)
    stopwords = {
        "qual", "quais", "como", "onde", "quando", "quem", "quanto", "quantos",
        "por", "para", "com", "sem", "que", "empresa", "documento", "resposta",
        "pergunta", "não", "tenho", "informações", "necessárias", "responder",
        "ano", "faturamento", "de", "do", "da", "dos", "das", "em", "no", "na",
        "nos", "nas", "foi", "é", "são", "reais", "milhões", "milhao", "mil"
    }
    sig_words = [t for t in tokens if len(t) >= 3 and t.lower() not in stopwords]

    scored_pages = []
    for doc, distance in results:
        page = doc.metadata.get("page", 0) + 1
        content = doc.page_content.lower()
        match_score = 0

        for num in sig_numbers:
            if num in doc.page_content:
                match_score += 15

        for word in sig_words:
            if word.lower() in content:
                match_score += 5

        scored_pages.append((page, match_score, distance))

    scored_pages.sort(key=lambda x: (-x[1], x[2]))

    best_score = scored_pages[0][1]
    if best_score > 0:
        exact_pages = sorted(list({p for p, s, d in scored_pages if s == best_score}))
        return exact_pages

    return [scored_pages[0][0]]


def ask_and_get_pages(query: str) -> Tuple[str, List[int]]:
    """
    Executa a busca RAG completa e retorna a resposta junto com a página exata
    onde a informação foi localizada no documento.
    """
    vector_store = get_vector_store()
    llm = get_llm()
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    try:
        results = vector_store.similarity_search_with_score(query, k=TOP_K)
    except Exception as e:
        err_str = str(e).lower()
        if "different vector dimensions" in err_str or "dataerror" in err_str:
            raise RuntimeError(
                f"\n⚠️ [INCOMPATIBILIDADE DE VETORES DETECTADA]\n"
                f"O banco de dados foi populado com outro modelo de embedding diferente do ativo ({AI_PROVIDER.upper()}: {ACTIVE_EMBEDDING_MODEL}).\n"
                f"Para sincronizar o banco com o novo modelo, execute no terminal:\n"
                f"  python src/ingest.py --force\n"
            ) from e
        raise e

    if not results:
        return FALLBACK_RESPONSE, []

    contexto = "\n\n".join([doc.page_content for doc, _score in results])
    chain = prompt | llm
    response = chain.invoke({"contexto": contexto, "pergunta": query})
    conteudo = response.content if hasattr(response, "content") else str(response)
    conteudo = conteudo.strip()

    exact_pages = locate_exact_pages(query, conteudo, results)
    return conteudo, exact_pages


def search_prompt(question: Optional[str] = None) -> Union[RunnableLambda, str, Any]:
    """
    Função oficial mantida para 100% de compatibilidade com os testes da Full Cycle.
    Retorna o Runnable ou a resposta direta caso question seja fornecido.
    """
    try:
        vector_store = get_vector_store()
        llm = get_llm()
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        def query_pipeline(query: str) -> str:
            contexto, _paginas, _score = search_context_and_pages(query, vector_store)
            chain = prompt | llm
            response = chain.invoke({"contexto": contexto, "pergunta": query})
            conteudo = response.content if hasattr(response, "content") else str(response)
            return conteudo.strip()

        pipeline_runnable = RunnableLambda(query_pipeline)

        if question is not None:
            return pipeline_runnable.invoke(question)

        return pipeline_runnable

    except Exception as e:
        print(f"Erro ao inicializar search_prompt: {e}")
        return None