import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Configurações centralizadas com validação Pydantic.
    Garante fail-fast caso alguma variável essencial esteja ausente.
    """
    GOOGLE_API_KEY: str = Field(default="", description="Chave de API do Google Gemini")
    GOOGLE_EMBEDDING_MODEL: str = Field(default="models/gemini-embedding-2", description="Modelo de embeddings")
    GOOGLE_CHAT_MODEL: str = Field(default="gemini-flash-lite-latest", description="Modelo LLM de geração")
    
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/rag",
        description="URL de conexão com o PostgreSQL"
    )
    PG_VECTOR_COLLECTION_NAME: str = Field(default="documentos_fullcycle", description="Nome da collection no pgVector")
    PDF_PATH: str = Field(default="./document.pdf", description="Caminho do PDF para ingestão")
    
    # Parâmetros oficiais exigidos pelo desafio Full Cycle
    CHUNK_SIZE: int = Field(default=1000, description="Tamanho de cada chunk de texto")
    CHUNK_OVERLAP: int = Field(default=150, description="Overlap entre chunks")
    TOP_K: int = Field(default=10, description="Quantidade de chunks mais relevantes")
    
    # Parâmetros avançados de MVP
    SCORE_THRESHOLD: float = Field(
        default=0.90,
        description="Limiar de distância máxima para corte de relevância (evita chamadas desnecessárias de API)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def normalized_database_url(self) -> str:
        """Normaliza a URL para o driver psycopg 3"""
        conn = self.DATABASE_URL
        if conn.startswith("postgresql://"):
            return conn.replace("postgresql://", "postgresql+psycopg://", 1)
        return conn


# Instância global reutilizável
settings = Settings()
