import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

# Suporte a UTF-8 no Windows para evitar erros de codificação de console (cp1252)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Parâmetros e variáveis de ambiente
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documentos_fullcycle")
PDF_PATH = os.getenv("PDF_PATH", "./document.pdf")
TOP_K = int(os.getenv("TOP_K", "10"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

from search import (
    search_prompt,
    ask_and_get_pages,
    get_vector_store,
    get_llm,
    AI_PROVIDER,
    ACTIVE_EMBEDDING_MODEL,
    ACTIVE_CHAT_MODEL,
    get_masked_key,
)

# Paleta cromática inspirada nos tons clássicos
C_GOLD = "#E5A823"        # Amarelo ocre / ouro luminoso
C_AMBER = "#C87D20"       # Âmbar aquecido
C_OCHRE = "#9E5318"       # Ocre terracota profundo
C_OLIVE = "#76874E"       # Verde-oliva vegetal
C_CREAM = "#F7E7B4"       # Creme palha suave
C_BORDER = "#D49B24"      # Contorno dourado encorpado

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.rule import Rule
    from rich import box
    USE_RICH = True
    console = Console(highlight=False)
except ImportError:
    USE_RICH = False
    console = None


def get_chunk_count() -> int:
    """Consulta o PostgreSQL de forma rápida para obter o total de chunks na collection."""
    try:
        import psycopg
        conn_str = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s;
                    """,
                    (PG_VECTOR_COLLECTION_NAME,),
                )
                res = cur.fetchone()
                return res[0] if res else 0
    except Exception:
        return 0


def print_banner():
    """Renderiza o banner de boas-vindas com visual estilizado e autoria."""
    provedor_label = "OpenAI" if AI_PROVIDER == "openai" else "Google Gemini"
    
    if USE_RICH:
        total_chunks = get_chunk_count()
        chunk_badge = f"{total_chunks} chunks indexados" if total_chunks > 0 else "Indexando..."

        banner_text = (
            f"[bold {C_CREAM}]Ingestão e Busca Semântica com LangChain e Postgres[/bold {C_CREAM}]\n"
            f"[{C_GOLD}]Desenvolvido por Breno Gomes Fernandes[/ {C_GOLD}] [dim {C_OCHRE}]•[/dim {C_OCHRE}] [{C_CREAM}]MBA em Engenharia de Software com IA[/ {C_CREAM}]\n\n"
            f"[dim {C_AMBER}]Provedor:[/dim {C_AMBER}] [bold {C_GOLD}]{provedor_label}[/bold {C_GOLD}]  [dim {C_OCHRE}]•[/dim {C_OCHRE}]  "
            f"[dim {C_AMBER}]Chave:[/dim {C_AMBER}] [{C_OLIVE}]{get_masked_key()}[/{C_OLIVE}]\n"
            f"[dim {C_AMBER}]LLM:[/dim {C_AMBER}] [bold {C_OLIVE}]{ACTIVE_CHAT_MODEL}[/bold {C_OLIVE}]  [dim {C_OCHRE}]•[/dim {C_OCHRE}]  "
            f"[dim {C_AMBER}]Embeddings:[/dim {C_AMBER}] [bold {C_OLIVE}]{ACTIVE_EMBEDDING_MODEL}[/bold {C_OLIVE}]\n"
            f"[dim {C_AMBER}]Base:[/dim {C_AMBER}] [bold {C_GOLD}]PostgreSQL + pgvector[/bold {C_GOLD}] ([{C_GOLD}]{chunk_badge}[/{C_GOLD}])  [dim {C_OCHRE}]•[/dim {C_OCHRE}]  "
            f"[dim {C_AMBER}]Top-K:[/dim {C_AMBER}] [bold {C_AMBER}]{TOP_K}[/bold {C_AMBER}]\n\n"
            f"[dim {C_OCHRE}]Comandos:[/dim {C_OCHRE}] "
            f"[bold {C_GOLD}]/info[/bold {C_GOLD}] [dim {C_OCHRE}]|[/dim {C_OCHRE}] "
            f"[bold {C_GOLD}]/limpar[/bold {C_GOLD}] [dim {C_OCHRE}]|[/dim {C_OCHRE}] "
            f"[bold {C_GOLD}]/sair[/bold {C_GOLD}]"
        )

        banner = Panel(
            banner_text,
            title=f"[bold {C_CREAM}]• Ingestão e Busca Semântica com LangChain e Postgres •[/bold {C_CREAM}]",
            border_style=C_BORDER,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        try:
            console.print()
            console.print(banner)
            console.print()
            return
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("  Ingestão e Busca Semântica com LangChain e Postgres")
    print("  Desenvolvedor: Breno Gomes Fernandes • MBA Full Cycle")
    print(f"  Provedor: {provedor_label} | Modelo: {ACTIVE_CHAT_MODEL}")
    print(f"  Embeddings: {ACTIVE_EMBEDDING_MODEL} | pgVector (k={TOP_K})")
    print("  Comandos: /info, /limpar, /sair")
    print("=" * 65 + "\n")


def show_system_info():
    """Exibe painel detalhado com parâmetros de ingestão, busca semântica e infraestrutura."""
    total_chunks = get_chunk_count()
    provedor_label = "OpenAI" if AI_PROVIDER == "openai" else "Google Gemini"

    if USE_RICH:
        table = Table(
            title=f"[bold {C_CREAM}]Parâmetros do Pipeline: Ingestão e Busca Semântica[/bold {C_CREAM}]",
            box=box.ROUNDED,
            header_style=f"bold {C_GOLD}",
            border_style=C_BORDER,
            show_header=True,
            padding=(0, 2),
        )
        table.add_column("Especificação", style=f"bold {C_CREAM}", width=28)
        table.add_column("Valor / Configuração Ativa", style=C_OLIVE)

        table.add_row(f"[{C_GOLD}]AUTORIA & PROJETO[/{C_GOLD}]", "")
        table.add_row("  Desenvolvedor", "Breno Gomes Fernandes")
        table.add_row("  Programa / MBA", "MBA em Engenharia de Software com IA (Full Cycle)")
        table.add_section()

        table.add_row(f"[{C_GOLD}]CONFIGURAÇÃO DE IA[/{C_GOLD}]", "")
        table.add_row("  Provedor Ativo", f"{provedor_label} (AI_PROVIDER={AI_PROVIDER})")
        table.add_row("  Chave de API", f"{get_masked_key()} (carregada do .env)")
        table.add_row("  Modelo LLM (Geração)", f"{ACTIVE_CHAT_MODEL} (temperatura: 0.0)")
        table.add_row("  Modelo de Embeddings", f"{ACTIVE_EMBEDDING_MODEL}")
        table.add_section()

        table.add_row(f"[{C_GOLD}]INFRAESTRUTURA & ARMAZENAMENTO[/{C_GOLD}]", "")
        table.add_row("  Banco de Dados", "PostgreSQL 17 + pgvector (localhost:5432/rag)")
        table.add_row("  Coleção no pgvector", f"{PG_VECTOR_COLLECTION_NAME} ({total_chunks} chunks)")
        table.add_row("  Driver / Conexão", "psycopg 3 (Pool Singleton ativo)")
        table.add_section()

        table.add_row(f"[{C_GOLD}]INGESTÃO DE DADOS[/{C_GOLD}]", "")
        table.add_row("  Documento Fonte", f"{PDF_PATH} (34 páginas)")
        table.add_row("  Divisão de Texto (Split)", f"{CHUNK_SIZE} caracteres por chunk (overlap: {CHUNK_OVERLAP})")
        table.add_row("  Estratégia de Ingestão", "Idempotente (SHA-256 + verificação de dimensão)")
        table.add_section()

        table.add_row(f"[{C_GOLD}]RECUPERAÇÃO SEMÂNTICA[/{C_GOLD}]", "")
        table.add_row("  Parâmetro de Busca", f"Top-{TOP_K} chunks mais relevantes (k={TOP_K})")
        table.add_row("  Formato de Saída", "Estrito (PERGUNTA / RESPOSTA)")

        console.print()
        console.print(table)
        console.print()
        return

    print("\n--- Parâmetros do Pipeline: Ingestão e Busca Semântica ---")
    print("Desenvolvedor: Breno Gomes Fernandes • MBA Full Cycle")
    print(f"Provedor Ativo: {provedor_label} ({AI_PROVIDER})")
    print(f"Chave de API: {get_masked_key()}")
    print(f"Modelo LLM: {ACTIVE_CHAT_MODEL} (temp: 0.0)")
    print(f"Modelo de Embeddings: {ACTIVE_EMBEDDING_MODEL}")
    print(f"Banco de Dados: PostgreSQL 17 + pgvector (localhost:5432/rag)")
    print(f"Coleção pgvector: {PG_VECTOR_COLLECTION_NAME} ({total_chunks} chunks)")
    print(f"Documento Fonte: {PDF_PATH} (34 páginas)")
    print(f"Segmentação: {CHUNK_SIZE} chars / overlap {CHUNK_OVERLAP}")
    print(f"Busca Semântica: Top-{TOP_K} chunks")
    print("----------------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Ingestão e Busca Semântica com LangChain e Postgres")
    parser.add_argument("-q", "--query", type=str, help="Executa uma consulta direta e encerra")
    args = parser.parse_args()

    # Pré-aquece conexões em background na inicialização
    try:
        get_vector_store()
        get_llm()
    except Exception:
        pass

    chain = search_prompt()
    if not chain:
        if USE_RICH:
            console.print(f"[bold {C_OCHRE}]Erro crítico:[/bold {C_OCHRE}] Não foi possível inicializar a cadeia de busca.")
        else:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    # Execução direta via flag --query (para testes e automação)
    if args.query:
        pergunta = args.query.strip()
        try:
            resposta, _paginas = ask_and_get_pages(pergunta)
            if USE_RICH:
                console.print(f"[bold {C_GOLD}]PERGUNTA:[/bold {C_GOLD}] {pergunta}")
                console.print(f"[bold {C_OLIVE}]RESPOSTA:[/bold {C_OLIVE}] {resposta}")
            else:
                print(f"PERGUNTA: {pergunta}")
                print(f"RESPOSTA: {resposta}")
        except Exception as e:
            print(f"Erro ao processar consulta: {e}")
        return

    print_banner()

    primeira_pergunta = True

    while True:
        try:
            if not primeira_pergunta and USE_RICH:
                console.print(Rule(style=C_BORDER))

            if USE_RICH:
                pergunta = console.input(f"[bold {C_GOLD}]PERGUNTA:[/bold {C_GOLD}] ").strip()
            else:
                pergunta = input("PERGUNTA: ").strip()

            if not pergunta:
                continue

            primeira_pergunta = False

            # Tratamento de comandos especiais
            comando = pergunta.lower()
            if comando in ["sair", "exit", "quit", "q", "/sair"]:
                if USE_RICH:
                    console.print(f"\n[bold {C_CREAM}]Encerrando assistente. Até logo![/bold {C_CREAM}]\n")
                else:
                    print("\nEncerrando...\n")
                break

            if comando in ["/limpar", "limpar", "clear", "cls"]:
                if USE_RICH:
                    console.clear()
                print_banner()
                primeira_pergunta = True
                continue

            if comando in ["/info", "info"]:
                show_system_info()
                continue

            # Processamento da consulta
            if USE_RICH:
                with console.status(f"[{C_GOLD}]Buscando no PostgreSQL e gerando resposta...[/{C_GOLD}]", spinner="dots"):
                    resposta, _paginas = ask_and_get_pages(pergunta)

                console.print(f"[bold {C_OLIVE}]RESPOSTA:[/bold {C_OLIVE}] {resposta}\n")
            else:
                print("[Buscando no banco e consultando IA...]\r", end="", flush=True)
                resposta, _paginas = ask_and_get_pages(pergunta)
                print(" " * 55 + "\r", end="", flush=True)

                print(f"RESPOSTA: {resposta}\n")

        except (KeyboardInterrupt, EOFError):
            if USE_RICH:
                console.print(f"\n[bold {C_CREAM}]Sessão finalizada pelo usuário.[/bold {C_CREAM}]\n")
            else:
                print("\nEncerrando...\n")
            break
        except Exception as e:
            if USE_RICH:
                console.print(f"\n[bold {C_OCHRE}]Erro ao processar pergunta:[/bold {C_OCHRE}] [{C_OCHRE}]{e}[/{C_OCHRE}]\n")
            else:
                print(f"\nErro ao processar pergunta: {e}\n")


if __name__ == "__main__":
    main()