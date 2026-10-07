import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "./document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "desafio_fullcycle")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-2")


def get_connection_string() -> str:
    conn = DATABASE_URL
    if conn.startswith("postgresql://"):
        conn = conn.replace("postgresql://", "postgresql+psycopg://", 1)
    return conn


def ingest_pdf():
    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Arquivo PDF não encontrado no caminho: {PDF_PATH}")

    print(f"1. Carregando documento: {pdf_file.name}...")
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    print(f"   Páginas carregadas: {len(documents)}")

    print("2. Dividindo o texto em chunks (chunk_size=1000, chunk_overlap=150)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   Total de chunks gerados: {len(chunks)}")

    print(f"3. Inicializando embeddings com {GOOGLE_EMBEDDING_MODEL}...")
    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)

    connection_string = get_connection_string()
    print(f"4. Conectando ao PostgreSQL (collection: {PG_VECTOR_COLLECTION_NAME})...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=connection_string,
        use_jsonb=True,
    )

    print("5. Armazenando vetores no banco de dados...")
    batch_size = 10
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   Processando lote {batch_num}/{total_batches} ({len(batch)} chunks)...")

        for attempt in range(5):
            try:
                vector_store.add_documents(batch)
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
    ingest_pdf()