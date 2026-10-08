import os
import sys
from typing import Tuple, List, Optional, Union, Any

# Suporte a execução de diferentes pontos de entrada
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

try:
    from src.config import settings
except ImportError:
    from config import settings


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
            model=settings.GOOGLE_EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        _CACHED_VECTOR_STORE = PGVector(
            embeddings=embeddings,
            collection_name=settings.PG_VECTOR_COLLECTION_NAME,
            connection=settings.normalized_database_url,
            use_jsonb=True,
        )
    return _CACHED_VECTOR_STORE


def get_llm() -> ChatGoogleGenerativeAI:
    """Inicializa uma única vez a LLM com temperatura zero."""
    global _CACHED_LLM
    if _CACHED_LLM is None:
        _CACHED_LLM = ChatGoogleGenerativeAI(
            model=settings.GOOGLE_CHAT_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0,
            transport="rest",
        )
    return _CACHED_LLM


def search_context_and_pages(query: str, vector_store: PGVector) -> Tuple[str, List[int], float]:
    """
    Recupera os k=10 chunks mais relevantes e extrai as páginas citadas.
    Retorna (contexto, lista_paginas_ordenadas, melhor_score).
    """
    results = vector_store.similarity_search_with_score(query, k=settings.TOP_K)
    
    if not results:
        return "", [], 1.0

    contexto = "\n\n".join([doc.page_content for doc, _score in results])
    
    # Extrai números únicos de páginas ordenadas para citação
    paginas = sorted(list({doc.metadata.get("page", 0) + 1 for doc, _score in results if "page" in doc.metadata}))
    
    melhor_score = min([score for _doc, score in results])
    return contexto, paginas, melhor_score


def ask_and_get_pages(query: str) -> Tuple[str, List[int]]:
    """
    Executa a busca RAG completa de forma robusta e estável.
    Retorna a resposta da LLM e a lista de páginas de referência.
    """
    vector_store = get_vector_store()
    llm = get_llm()
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    contexto, paginas, _score = search_context_and_pages(query, vector_store)
    chain = prompt | llm
    response = chain.invoke({"contexto": contexto, "pergunta": query})
    conteudo = response.content if hasattr(response, "content") else str(response)
    return conteudo.strip(), paginas


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