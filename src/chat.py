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
    """Renderiza o banner de boas-vindas com visual moderno e profissional."""
    provedor_label = "OpenAI" if AI_PROVIDER == "openai" else "Google Gemini"
    
    if USE_RICH:
        total_chunks = get_chunk_count()
        chunk_badge = f"{total_chunks} chunks indexados" if total_chunks > 0 else "Indexando..."

        banner_text = (
            "[bold white]Ingestão e Busca Semântica com LangChain e Postgres[/bold white]\n"
            "[cyan]Desafio MBA • Engenharia de Software com IA[/cyan]\n\n"
            f"[dim]Provedor:[/dim] [bold cyan]{provedor_label}[/bold cyan]  •  "
            f"[dim]Chave:[/dim] [dim green]{get_masked_key()}[/dim green]\n"
            f"[dim]LLM:[/dim] [bold green]{ACTIVE_CHAT_MODEL}[/bold green]  •  "
            f"[dim]Embeddings:[/dim] [bold green]{ACTIVE_EMBEDDING_MODEL}[/bold green]\n"
            f"[dim]Base:[/dim] [bold yellow]PostgreSQL + pgvector[/bold yellow] ([yellow]{chunk_badge}[/yellow])  •  "
            f"[dim]Top-K:[/dim] [bold magenta]{TOP_K}[/bold magenta]\n\n"
            "[dim]Comandos:[/dim] "
            "[bold cyan]/info[/bold cyan] [dim]|[/dim] "
            "[bold cyan]/limpar[/bold cyan] [dim]|[/dim] "
            "[bold cyan]/sair[/bold cyan]"
        )

        banner = Panel(
            banner_text,
            title="[bold cyan]• Ingestão e Busca Semântica com LangChain e Postgres •[/bold cyan]",
            border_style="cyan",
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
            title="[bold cyan]Parâmetros do Pipeline: Ingestão e Busca Semântica[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="dim cyan",
            show_header=True,
            padding=(0, 2),
        )
        table.add_column("Especificação", style="bold white", width=28)
        table.add_column("Valor / Configuração Ativa", style="green")

        table.add_row("[yellow]CONFIGURAÇÃO DE IA[/yellow]", "")
        table.add_row("  Provedor Ativo", f"{provedor_label} (AI_PROVIDER={AI_PROVIDER})")
        table.add_row("  Chave de API", f"{get_masked_key()} (carregada do .env)")
        table.add_row("  Modelo LLM (Geração)", f"{ACTIVE_CHAT_MODEL} (temperatura: 0.0)")
        table.add_row("  Modelo de Embeddings", f"{ACTIVE_EMBEDDING_MODEL}")
        table.add_section()

        table.add_row("[yellow]INFRAESTRUTURA & ARMAZENAMENTO[/yellow]", "")
        table.add_row("  Banco de Dados", "PostgreSQL 17 + pgvector (localhost:5432/rag)")
        table.add_row("  Coleção no pgvector", f"{PG_VECTOR_COLLECTION_NAME} ({total_chunks} chunks)")
        table.add_row("  Driver / Conexão", "psycopg 3 (Pool Singleton ativo)")
        table.add_section()

        table.add_row("[yellow]INGESTÃO DE DADOS[/yellow]", "")
        table.add_row("  Documento Fonte", f"{PDF_PATH} (34 páginas)")
        table.add_row("  Divisão de Texto (Split)", f"{CHUNK_SIZE} caracteres por chunk (overlap: {CHUNK_OVERLAP})")
        table.add_row("  Estratégia de Ingestão", "Idempotente (SHA-256 + verificação de dimensão)")
        table.add_section()

        table.add_row("[yellow]RECUPERAÇÃO SEMÂNTICA[/yellow]", "")
        table.add_row("  Parâmetro de Busca", f"Top-{TOP_K} chunks mais relevantes (k={TOP_K})")
        table.add_row("  Formato de Saída", "Estrito (PERGUNTA / RESPOSTA)")

        console.print()
        console.print(table)
        console.print()
        return

    print("\n--- Parâmetros do Pipeline: Ingestão e Busca Semântica ---")
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
            console.print("[bold red]Erro crítico:[/bold red] Não foi possível inicializar a cadeia de busca.")
        else:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    # Execução direta via flag --query (para testes e automação)
    if args.query:
        pergunta = args.query.strip()
        try:
            resposta, _paginas = ask_and_get_pages(pergunta)
            if USE_RICH:
                console.print(f"[bold yellow]PERGUNTA:[/bold yellow] {pergunta}")
                console.print(f"[bold green]RESPOSTA:[/bold green] {resposta}")
            else:
                print(f"PERGUNTA: {pergunta}")
                print(f"RESPOSTA: {resposta}")
        except RuntimeError as re_err:
            print(str(re_err))
        except Exception as e:
            print(f"Erro ao processar consulta: {e}")
        return

    print_banner()

    primeira_pergunta = True

    while True:
        try:
            if not primeira_pergunta and USE_RICH:
                console.print(Rule(style="dim cyan"))

            if USE_RICH:
                pergunta = console.input("[bold yellow]PERGUNTA:[/bold yellow] ").strip()
            else:
                pergunta = input("PERGUNTA: ").strip()

            if not pergunta:
                continue

            primeira_pergunta = False

            # Tratamento de comandos especiais
            comando = pergunta.lower()
            if comando in ["sair", "exit", "quit", "q", "/sair"]:
                if USE_RICH:
                    console.print("\n[bold cyan]Encerrando assistente. Até logo![/bold cyan]\n")
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
                with console.status("[bold cyan]Buscando no PostgreSQL e gerando resposta...[/bold cyan]", spinner="dots"):
                    resposta, _paginas = ask_and_get_pages(pergunta)

                console.print(f"[bold green]RESPOSTA:[/bold green] {resposta}\n")
            else:
                print("[Buscando no banco e consultando IA...]\r", end="", flush=True)
                resposta, _paginas = ask_and_get_pages(pergunta)
                print(" " * 55 + "\r", end="", flush=True)

                print(f"RESPOSTA: {resposta}\n")

        except (KeyboardInterrupt, EOFError):
            if USE_RICH:
                console.print("\n[bold cyan]Sessão finalizada pelo usuário.[/bold cyan]\n")
            else:
                print("\nEncerrando...\n")
            break
        except RuntimeError as re_err:
            if USE_RICH:
                console.print(f"[bold yellow]{re_err}[/bold yellow]\n")
            else:
                print(f"{re_err}\n")
        except Exception as e:
            if USE_RICH:
                console.print(f"\n[bold red]Erro ao processar pergunta:[/bold red] [red]{e}[/red]\n")
            else:
                print(f"\nErro ao processar pergunta: {e}\n")


if __name__ == "__main__":
    main()