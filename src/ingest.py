import os
import sys
import time
import hashlib
from pathlib import Path
import psycopg

# Suporte a execução tanto via 'python src/ingest.py' quanto 'python -m src.ingest'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

try:
    from src.config import settings
except ImportError:
    from config import settings


def calculate_file_hash(file_path: Path) -> str:
    """Calcula o hash SHA-256 do arquivo para garantir idempotência e rastreabilidade."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_existing_chunks(file_hash: str) -> int:
    """Consulta o banco de dados para verificar se o documento já foi ingerido."""
    try:
        conn_str = settings.normalized_database_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*) 
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s AND e.cmetadata->>'doc_hash' = %s;
                    """,
                    (settings.PG_VECTOR_COLLECTION_NAME, file_hash),
                )
                res = cur.fetchone()
                return res[0] if res else 0
    except Exception:
        # Se as tabelas ainda não existirem (primeira execução), retorna 0
        return 0


def delete_existing_chunks():
    """Limpa os registros da collection atual para re-ingestão limpa."""
    try:
        conn_str = settings.normalized_database_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding e
                    USING langchain_pg_collection c
                    WHERE e.collection_id = c.uuid AND c.name = %s;
                    """,
                    (settings.PG_VECTOR_COLLECTION_NAME,),
                )
    except Exception:
        pass


def ingest_pdf(force_reset: bool = False):
    """
    Executa a ingestão do documento PDF de forma 100% idempotente:
    - Fatiamento nos padrões Full Cycle (chunk_size=1000, chunk_overlap=150)
    - Verificação de documento pré-existente (não consome tokens de API se já foi ingerido)
    - IDs determinísticos que garantem que nunca haverá duplicações no PostgreSQL
    """
    pdf_file = Path(settings.PDF_PATH)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Arquivo PDF não encontrado no caminho: {settings.PDF_PATH}")

    file_hash = calculate_file_hash(pdf_file)
    print(f"1. Carregando documento: {pdf_file.name} (SHA-256: {file_hash[:12]}...)...")

    # Verificação de idempotência prévia:
    existing_count = count_existing_chunks(file_hash)
    if existing_count > 0 and not force_reset:
        print(f"\n[IDEMPOTÊNCIA ATIVA] O documento '{pdf_file.name}' já está gravado no banco ({existing_count} chunks encontrados).")
        print("Nenhum embedding novo precisa ser gerado (Economia de 100% da cota da sua API).")
        print("Para forçar a reinserção do zero, use: python src/ingest.py --force\n")
        return

    if force_reset:
        print("   Limpando dados anteriores para re-ingestão forçada...")
        delete_existing_chunks()

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    print(f"   Páginas carregadas: {len(documents)}")

    print(f"2. Dividindo o texto em chunks (chunk_size={settings.CHUNK_SIZE}, chunk_overlap={settings.CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
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
        
        # ID determinístico único por hash e posição do chunk
        chunk_id = f"{file_hash[:10]}_p{page_num:03d}_c{idx:04d}"
        chunk_ids.append(chunk_id)

    print(f"3. Inicializando embeddings com {settings.GOOGLE_EMBEDDING_MODEL}...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GOOGLE_EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    print(f"4. Conectando ao PostgreSQL (collection: {settings.PG_VECTOR_COLLECTION_NAME})...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=settings.PG_VECTOR_COLLECTION_NAME,
        connection=settings.normalized_database_url,
        use_jsonb=True,
    )

    print("5. Armazenando vetores no banco de dados...")
    batch_size = 10
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = chunk_ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   Processando lote {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")

        for attempt in range(5):
            try:
                vector_store.add_documents(batch_chunks, ids=batch_ids)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    wait_time = (attempt + 1) * 3
                    print(f"   Rate limit atingido. Aguardando {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        time.sleep(1)

    print("Ingestão concluída com sucesso no PostgreSQL + pgVector!")


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest_pdf(force_reset=force)