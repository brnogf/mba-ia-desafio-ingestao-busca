import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Optional
import psycopg
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
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL") or "models/gemini-embedding-001"
ACTIVE_EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL if AI_PROVIDER == "openai" else GOOGLE_EMBEDDING_MODEL

# Parâmetros de Banco de Dados e RAG
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME") or "documentos_fullcycle"
PDF_PATH = os.getenv("PDF_PATH") or "./document.pdf"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or "1000")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or "150")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings


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


def calculate_file_hash(file_path: Path) -> str:
    """Calcula o hash SHA-256 do arquivo para garantir idempotência e rastreabilidade."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_stored_vector_dims() -> Optional[int]:
    """Consulta o banco de dados para verificar a dimensão dos vetores já gravados."""
    try:
        conn_str = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT vector_dims(e.embedding)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    LIMIT 1;
                    """,
                    (PG_VECTOR_COLLECTION_NAME,),
                )
                res = cur.fetchone()
                return res[0] if res else None
    except Exception:
        return None


def count_existing_chunks(file_hash: str) -> int:
    """Consulta o banco de dados para verificar se o documento já foi ingerido."""
    try:
        conn_str = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*) 
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s AND e.cmetadata->>'doc_hash' = %s;
                    """,
                    (PG_VECTOR_COLLECTION_NAME, file_hash),
                )
                res = cur.fetchone()
                return res[0] if res else 0
    except Exception:
        return 0


def delete_existing_chunks():
    """Limpa os registros da collection atual para re-ingestão limpa."""
    try:
        conn_str = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding e
                    USING langchain_pg_collection c
                    WHERE e.collection_id = c.uuid AND c.name = %s;
                    """,
                    (PG_VECTOR_COLLECTION_NAME,),
                )
    except Exception:
        pass


def ingest_pdf(force_reset: bool = False):
    """
    Executa a ingestão do documento PDF de forma 100% idempotente e auto-recuperável:
    - Fatiamento nos padrões Full Cycle (chunk_size=1000, chunk_overlap=150)
    - Suporte dual transparente para OpenAI e Google Gemini
    - Auto-detecção de incompatibilidade de dimensões vetoriais no PostgreSQL
    - Sincronização automática quando detectada troca de API key / modelo de embeddings
    """
    pdf_file = Path(PDF_PATH)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"Arquivo PDF não encontrado no caminho: {PDF_PATH}")

    file_hash = calculate_file_hash(pdf_file)
    print(f"1. Carregando documento: {pdf_file.name} (SHA-256: {file_hash[:12]}...)...")

    embeddings = get_embeddings()

    # Verificação de compatibilidade de dimensão entre o banco e o modelo ativo
    stored_dims = get_stored_vector_dims()
    existing_count = count_existing_chunks(file_hash)

    if stored_dims is not None and existing_count > 0:
        expected_dims = 1536 if AI_PROVIDER == "openai" else 3072
        if (AI_PROVIDER == "openai" and stored_dims != 1536) or (AI_PROVIDER == "gemini" and stored_dims not in (768, 3072)):
            print(f"\n⚠️ Detectada mudança de API key / modelo de embeddings ({stored_dims} -> {expected_dims} dimensões).")
            print("   Sincronizando e atualizando o banco de dados automaticamente, por favor aguarde...\n")
            force_reset = True

    # Verificação de idempotência prévia:
    if existing_count > 0 and not force_reset:
        print(f"\n[IDEMPOTÊNCIA ATIVA] O documento '{pdf_file.name}' já está gravado no banco ({existing_count} chunks encontrados).")
        print(f"Provedor ativo: {AI_PROVIDER.upper()} | Modelo: {ACTIVE_EMBEDDING_MODEL}")
        print("Nenhum embedding novo precisa ser gerado (Economia de 100% da cota da sua API).\n")
        return

    if force_reset:
        print("   Atualizando dados no banco para o novo modelo de embeddings...")
        delete_existing_chunks()

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    print(f"   Páginas carregadas: {len(documents)}")

    print(f"2. Dividindo o texto em chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   Total de chunks gerados: {len(chunks)}")

    # Enriquecimento com metadados para auditoria e IDs determinísticos
    chunk_ids = []
    for idx, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", 0) + 1
        chunk.metadata["arquivo_nome"] = pdf_file.name
        chunk.metadata["pagina"] = page_num
        chunk.metadata["doc_hash"] = file_hash
        chunk.metadata["chunk_index"] = idx
        
        chunk_id = f"{file_hash[:10]}_p{page_num:03d}_c{idx:04d}"
        chunk_ids.append(chunk_id)

    print(f"3. Inicializando embeddings [{AI_PROVIDER.upper()}] com {ACTIVE_EMBEDDING_MODEL}...")

    print(f"4. Conectando ao PostgreSQL (collection: {PG_VECTOR_COLLECTION_NAME})...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    print("5. Armazenando vetores no banco de dados...")
    if AI_PROVIDER == "gemini":
        batch_size = 5
        sleep_between_batches = 20
    else:
        batch_size = 20
        sleep_between_batches = 1

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    from tqdm import tqdm
    
    with tqdm(total=total_batches, desc="Processando lotes", unit="lote") as pbar:
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_ids = chunk_ids[i : i + batch_size]

            for attempt in range(3):
                try:
                    vector_store.add_documents(batch_chunks, ids=batch_ids)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait_time = (attempt + 1) * 30
                        pbar.write(f"   Rate limit atingido (429). Aguardando {wait_time}s para reset de cota...")
                        time.sleep(wait_time)
                    else:
                        raise e
                        
            if i + batch_size < len(chunks):
                time.sleep(sleep_between_batches)
            pbar.update(1)

    print("\n✅ Ingestão concluída com sucesso no PostgreSQL + pgVector!")


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest_pdf(force_reset=force)