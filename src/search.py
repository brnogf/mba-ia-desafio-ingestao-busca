import os
import sys
import re
from typing import Tuple, List, Optional, Union, Any
from dotenv import load_dotenv

load_dotenv()

# Parâmetros e variáveis de ambiente
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ ERRO: A variável GOOGLE_API_KEY não foi preenchida no arquivo .env!")

GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL") or "models/text-embedding-004"
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-flash-lite-latest")
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documentos_fullcycle")
TOP_K = int(os.getenv("TOP_K", "10"))

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector


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
_CACHED_LLM: Optional[ChatGoogleGenerativeAI] = None


def get_vector_store() -> PGVector:
    """Inicializa uma única vez e reutiliza o pool de conexões do PostgreSQL."""
    global _CACHED_VECTOR_STORE
    if _CACHED_VECTOR_STORE is None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        _CACHED_VECTOR_STORE = PGVector(
            embeddings=embeddings,
            collection_name=PG_VECTOR_COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
    return _CACHED_VECTOR_STORE


def get_llm() -> ChatGoogleGenerativeAI:
    """Inicializa uma única vez a LLM com temperatura zero."""
    global _CACHED_LLM
    if _CACHED_LLM is None:
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
    Retorna (contexto, lista_paginas_ordenadas, melhor_score).
    """
    results = vector_store.similarity_search_with_score(query, k=TOP_K)
    
    if not results:
        return "", [], 1.0

    contexto = "\n\n".join([doc.page_content for doc, _score in results])
    paginas = sorted(list({doc.metadata.get("page", 0) + 1 for doc, _score in results if "page" in doc.metadata}))
    melhor_score = min([score for _doc, score in results])
    return contexto, paginas, melhor_score


def locate_exact_pages(query: str, answer: str, results: List[Tuple[Any, float]]) -> List[int]:
    """
    Identifica com precisão cirúrgica a página do documento onde a informação
    que respondeu à pergunta realmente está localizada, em vez de listar todas
    as páginas recuperadas pelo Top-K.
    """
    if not results or FALLBACK_RESPONSE in answer:
        return []

    # Extrai números, valores monetários e anos da resposta
    numbers = re.findall(r"[\d\.,]+", answer)
    sig_numbers = [n.strip(".,") for n in numbers if len(n.strip(".,")) >= 2]

    # Extrai termos e entidades relevantes da pergunta e da resposta
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

        # Correspondência de números/valores da resposta (peso altíssimo)
        for num in sig_numbers:
            if num in doc.page_content:
                match_score += 15

        # Correspondência de termos-chave e nomes de entidades
        for word in sig_words:
            if word.lower() in content:
                match_score += 5

        scored_pages.append((page, match_score, distance))

    scored_pages.sort(key=lambda x: (-x[1], x[2]))

    best_score = scored_pages[0][1]
    if best_score > 0:
        exact_pages = sorted(list({p for p, s, d in scored_pages if s == best_score}))
        return exact_pages

    # Fallback para o chunk de menor distância vetorial
    return [scored_pages[0][0]]


def ask_and_get_pages(query: str) -> Tuple[str, List[int]]:
    """
    Executa a busca RAG completa e retorna a resposta junto com a página exata
    onde a informação foi localizada no documento.
    """
    vector_store = get_vector_store()
    llm = get_llm()
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    results = vector_store.similarity_search_with_score(query, k=TOP_K)
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