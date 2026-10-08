import os
import sys
import time

# Suporte a UTF-8 no Windows para evitar erros de codificação de console (cp1252)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Garante acesso à raiz do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import (
    search_prompt,
    ask_and_get_pages,
    get_vector_store,
    get_llm,
)

try:
    from src.config import settings
except ImportError:
    from config import settings

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
        conn_str = settings.normalized_database_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s;
                    """,
                    (settings.PG_VECTOR_COLLECTION_NAME,),
                )
                res = cur.fetchone()
                return res[0] if res else 0
    except Exception:
        return 0


def print_banner():
    """Renderiza o banner de boas-vindas com visual moderno e profissional."""
    if USE_RICH:
        total_chunks = get_chunk_count()
        chunk_badge = f"{total_chunks} chunks indexados" if total_chunks > 0 else "Indexando..."

        banner_text = (
            "[bold white]FULL CYCLE MBA • ENGENHARIA DE SOFTWARE COM IA[/bold white]\n"
            "[cyan]Assistente de Busca Semântica e RAG com PostgreSQL (pgvector)[/cyan]\n\n"
            f"[dim]LLM:[/dim] [bold green]{settings.GOOGLE_CHAT_MODEL}[/bold green]  •  "
            f"[dim]Embeddings:[/dim] [bold green]{settings.GOOGLE_EMBEDDING_MODEL}[/bold green]\n"
            f"[dim]Base:[/dim] [bold yellow]PostgreSQL + pgvector[/bold yellow] ([yellow]{chunk_badge}[/yellow])  •  "
            f"[dim]Top-K:[/dim] [bold magenta]{settings.TOP_K}[/bold magenta]\n\n"
            "[dim]Comandos:[/dim] [bold cyan]/ajuda[/bold cyan] [dim]|[/dim] "
            "[bold cyan]/info[/bold cyan] [dim]|[/dim] "
            "[bold cyan]/limpar[/bold cyan] [dim]|[/dim] "
            "[bold cyan]/sair[/bold cyan]"
        )

        banner = Panel(
            banner_text,
            title="[bold cyan]• RAG Terminal Assistant •[/bold cyan]",
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

    print("\n" + "=" * 55)
    print("  FULL CYCLE MBA - CHATBOT RAG (MVP)")
    print(f"  Modelo: {settings.GOOGLE_CHAT_MODEL} | pgVector (k={settings.TOP_K})")
    print("  Comandos: /ajuda, /info, /limpar, /sair")
    print("=" * 55 + "\n")


def show_system_info():
    """Exibe painel detalhado de diagnóstico da infraestrutura e parâmetros."""
    total_chunks = get_chunk_count()

    if USE_RICH:
        table = Table(
            title="[bold cyan]Diagnóstico Técnico & Parâmetros do Sistema (RAG)[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="dim cyan",
            show_header=True,
            padding=(0, 2),
        )
        table.add_column("Especificação", style="bold white", width=28)
        table.add_column("Valor / Configuração Ativa", style="green")

        table.add_row("[yellow]INFRAESTRUTURA[/yellow]", "")
        table.add_row("  Banco de Dados", "PostgreSQL 17 + pgvector (localhost:5432/rag)")
        table.add_row("  Coleção no pgvector", f"{settings.PG_VECTOR_COLLECTION_NAME} ({total_chunks} chunks armazenados)")
        table.add_row("  Driver / Conexão", "psycopg 3 (Pool Singleton ativo)")
        table.add_section()

        table.add_row("[yellow]INGESTÃO DE DADOS[/yellow]", "")
        table.add_row("  Documento Fonte", f"{settings.PDF_PATH} (34 páginas, ~175 KB)")
        table.add_row("  Divisão de Texto (Split)", f"{settings.CHUNK_SIZE} caracteres por chunk (overlap: {settings.CHUNK_OVERLAP})")
        table.add_row("  Estratégia de Ingestão", "Idempotente (SHA-256 verificado)")
        table.add_section()

        table.add_row("[yellow]INTELIGÊNCIA ARTIFICIAL[/yellow]", "")
        table.add_row("  Modelo de Embeddings", f"{settings.GOOGLE_EMBEDDING_MODEL} (768 dimensões)")
        table.add_row("  Modelo LLM (Geração)", f"{settings.GOOGLE_CHAT_MODEL} (temperatura: 0.0)")
        table.add_row("  Recuperação Semântica", f"Top-{settings.TOP_K} chunks mais relevantes (k={settings.TOP_K})")
        table.add_row("  Transporte / Latência", "REST API direta (~1.2s - 2.0s por resposta)")

        console.print()
        console.print(table)
        console.print()
        return

    print("\n--- Diagnóstico Técnico (RAG) ---")
    print(f"Banco de Dados: PostgreSQL 17 + pgvector (localhost:5432/rag)")
    print(f"Coleção pgvector: {settings.PG_VECTOR_COLLECTION_NAME} ({total_chunks} chunks)")
    print(f"Documento Fonte: {settings.PDF_PATH} (34 páginas)")
    print(f"Segmentação: {settings.CHUNK_SIZE} chars / overlap {settings.CHUNK_OVERLAP}")
    print(f"Embeddings: {settings.GOOGLE_EMBEDDING_MODEL}")
    print(f"Modelo LLM: {settings.GOOGLE_CHAT_MODEL} (temp: 0.0)")
    print(f"Busca Semântica: Top-{settings.TOP_K} chunks")
    print("---------------------------------\n")


def show_help():
    """Exibe guia rápido com comandos, regras de RAG e exemplos de perguntas."""
    if USE_RICH:
        table = Table(
            title="[bold cyan]Central de Ajuda • Comandos & Guia de Perguntas[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="dim cyan",
            show_header=True,
            padding=(0, 2),
        )
        table.add_column("Item", style="bold white", width=22)
        table.add_column("Orientação / Exemplo Prático", style="white")

        table.add_row("[yellow]COMANDOS DO CLI[/yellow]", "")
        table.add_row("  /info", "Exibe diagnóstico técnico da infraestrutura e modelos")
        table.add_row("  /limpar", "Limpa o terminal e restaura a visualização inicial")
        table.add_row("  /ajuda", "Exibe esta referência de comandos e boas práticas")
        table.add_row("  /sair", "Encerra o assistente de forma limpa")
        table.add_section()

        table.add_row("[yellow]DIRETRIZES DE RAG[/yellow]", "")
        table.add_row("  Dado Explícito", "Cite o nome exato da empresa pesquisada (ex: SuperTechIABrazil)")
        table.add_row("  Busca Posicional", "Evite termos como primeira empresa (chunks são recuperados por similaridade)")
        table.add_row("  Fallback Estrito", "Perguntas fora do contexto retornam mensagem padrão sem alucinações")
        table.add_section()

        table.add_row("[yellow]EXEMPLOS DE TESTE[/yellow]", "")
        table.add_row("  Consulta no PDF", "[cyan]Qual o faturamento da Empresa SuperTechIABrazil?[/cyan] (Exemplo oficial)")
        table.add_row("  Consulta no PDF", "[cyan]Qual o faturamento da empresa Alfa Agronegócio Indústria?[/cyan]")
        table.add_row("  Metadado no PDF", "[cyan]Em que ano foi fundada a empresa Alfa Energia S.A.?[/cyan]")
        table.add_row("  Fora de Escopo", "[dim]Quantos clientes temos em 2024?[/dim] [yellow](Testa o fallback)[/yellow]")

        console.print()
        console.print(table)
        console.print()
        return

    print("\n--- Central de Ajuda & Diretrizes ---")
    print("Comandos: /info, /limpar, /ajuda, /sair")
    print("Diretrizes de busca:")
    print(" - Cite o nome exato da empresa (ex: SuperTechIABrazil)")
    print(" - Evite buscas posicionais como 'primeira empresa'")
    print("Exemplos:")
    print(" - Qual o faturamento da Empresa SuperTechIABrazil?")
    print(" - Quantos clientes temos em 2024? (teste de fallback)")
    print("------------------------------------\n")


def main():
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

            if comando in ["/ajuda", "ajuda", "help", "/help", "?"]:
                show_help()
                continue

            # Processamento da consulta com medição de latência
            t0 = time.time()
            resposta = ""
            paginas = []

            if USE_RICH:
                with console.status("[bold cyan]Buscando no PostgreSQL e gerando resposta...[/bold cyan]", spinner="dots"):
                    resposta, paginas = ask_and_get_pages(pergunta)
                elapsed = time.time() - t0

                # Formato obrigatório para testes: PERGUNTA e RESPOSTA
                console.print(f"[bold green]RESPOSTA:[/bold green] {resposta}")

                # Rodapé informando a página exata da informação
                if "Não tenho informações necessárias" not in resposta and paginas:
                    paginas_str = ", ".join(str(p) for p in paginas)
                    label = "Página da informação:" if len(paginas) == 1 else "Páginas da informação:"
                    console.print(
                        f"[dim cyan]{label}[/dim cyan] [bold cyan]{paginas_str}[/bold cyan] "
                        f"[dim]({settings.PDF_PATH}) • Tempo:[/dim] [dim green]{elapsed:.2f}s[/dim green]\n"
                    )
                else:
                    console.print(
                        f"[dim yellow][Info][/dim yellow] [dim]Informação não encontrada no documento. "
                        f"• Tempo: {elapsed:.2f}s[/dim]\n"
                    )
            else:
                print("[Buscando no banco e consultando IA...]\r", end="", flush=True)
                resposta, paginas = ask_and_get_pages(pergunta)
                elapsed = time.time() - t0
                print(" " * 55 + "\r", end="", flush=True)

                print(f"RESPOSTA: {resposta}")
                if "Não tenho informações necessárias" not in resposta and paginas:
                    paginas_str = ", ".join(str(p) for p in paginas)
                    label = "Página da informação" if len(paginas) == 1 else "Páginas da informação"
                    print(f"[{label}: {paginas_str} ({settings.PDF_PATH}) - {elapsed:.2f}s]\n")
                else:
                    print(f"[Tempo: {elapsed:.2f}s]\n")

        except (KeyboardInterrupt, EOFError):
            if USE_RICH:
                console.print("\n[bold cyan]Sessão finalizada pelo usuário.[/bold cyan]\n")
            else:
                print("\nEncerrando...\n")
            break
        except Exception as e:
            if USE_RICH:
                console.print(f"\n[bold red]Erro ao processar pergunta:[/bold red] [red]{e}[/red]\n")
            else:
                print(f"\nErro ao processar pergunta: {e}\n")


if __name__ == "__main__":
    main()