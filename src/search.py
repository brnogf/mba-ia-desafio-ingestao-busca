import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "desafio_fullcycle")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-2")
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.8-flash")

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


def get_connection_string() -> str:
    conn = DATABASE_URL
    if conn.startswith("postgresql://"):
        conn = conn.replace("postgresql://", "postgresql+psycopg://", 1)
    return conn


def search_prompt(question=None):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
        connection_string = get_connection_string()

        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=PG_VECTOR_COLLECTION_NAME,
            connection=connection_string,
            use_jsonb=True,
        )

        llm = ChatGoogleGenerativeAI(
            model=GOOGLE_CHAT_MODEL,
            temperature=0.0,
        )

        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        def query_pipeline(query: str) -> str:
            # 1. Vetorizar a pergunta e buscar os 10 resultados mais relevantes (k=10)
            results = vector_store.similarity_search_with_score(query, k=10)
            
            # 2. Concatenar resultados como contexto
            contexto = "\n\n".join([doc.page_content for doc, _score in results])
            
            # 3. Executar o prompt com a LLM
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