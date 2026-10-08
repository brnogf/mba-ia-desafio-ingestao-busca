import os
import sys

# Suporte a UTF-8 no Windows para evitar erros de codificação cp1252
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
    search_prompt_stream,
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
    USE_RICH = True
    console = Console(highlight=False)
except ImportError:
    USE_RICH = False


def print_banner():
    if USE_RICH:
        banner = Panel.fit(
            "[bold cyan]Full Cycle RAG - Engenharia de Software com IA[/bold cyan]\n"
            f"[dim]Modelo:[/dim] [green]{settings.GOOGLE_CHAT_MODEL}[/green] | "
            f"[dim]Embeddings:[/dim] [green]{settings.GOOGLE_EMBEDDING_MODEL}[/green] | "
            f"[dim]Banco:[/dim] [yellow]PostgreSQL + pgVector[/yellow]\n\n"
            "[white]Comandos especiais:[/white] [cyan]/info[/cyan] (status do sistema) | [cyan]/sair[/cyan] (encerrar)",
            title="Assistente Semantico (MVP)",
            border_style="cyan",
        )
        try:
            console.print(banner)
            console.print()
            return
        except Exception:
            pass

    print("\n=== Chatbot RAG Full Cycle (MVP) ===")
    print("Digite sua pergunta ou 'sair' para encerrar.\n")


def show_system_info():
    print(f"\n--- Informações do Sistema ---")
    print(f"Documento Base: {settings.PDF_PATH}")
    print(f"Collection no PostgreSQL: {settings.PG_VECTOR_COLLECTION_NAME}")
    print(f"Modelo LLM: {settings.GOOGLE_CHAT_MODEL}")
    print(f"Modelo Embeddings: {settings.GOOGLE_EMBEDDING_MODEL}")
    print(f"Chunks recuperados por busca (k): {settings.TOP_K}")
    print(f"-------------------------------\n")


def main():
    # Pré-aquece conexões em background na inicialização para acelerar a 1ª pergunta
    try:
        get_vector_store()
        get_llm()
    except Exception:
        pass

    chain = search_prompt()
    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print_banner()

    while True:
        try:
            if USE_RICH:
                pergunta = console.input("[bold yellow]PERGUNTA:[/bold yellow] ").strip()
            else:
                pergunta = input("PERGUNTA: ").strip()

            if not pergunta:
                continue

            if pergunta.lower() in ["sair", "exit", "quit", "q", "/sair"]:
                print("Encerrando...")
                break

            if pergunta.lower() in ["/info", "info"]:
                show_system_info()
                continue

            if pergunta.lower() in ["/ajuda", "ajuda", "help"]:
                print("\nDigite qualquer pergunta sobre o documento fornecido ou 'sair' para sair.\n")
                continue

            # Feedback visual instantâneo durante a busca e o tempo até o primeiro token
            first_token = ""
            tokens_stream = None
            paginas = []

            if USE_RICH:
                with console.status("[cyan]Pesquisando no PostgreSQL e consultando IA...[/cyan]", spinner="dots"):
                    tokens_stream, paginas = search_prompt_stream(pergunta)
                    try:
                        first_token = next(tokens_stream)
                    except StopIteration:
                        first_token = ""

                console.print("[bold green]RESPOSTA:[/bold green] ", end="")
            else:
                print("[Buscando no banco e consultando IA...]\r", end="", flush=True)
                tokens_stream, paginas = search_prompt_stream(pergunta)
                try:
                    first_token = next(tokens_stream)
                except StopIteration:
                    first_token = ""
                print(" " * 50 + "\r", end="", flush=True)  # Limpa mensagem de busca
                print("RESPOSTA: ", end="", flush=True)

            # Imprime o primeiro token e continua com o streaming em tempo real
            resposta_completa = first_token
            print(first_token, end="", flush=True)

            if tokens_stream:
                for token in tokens_stream:
                    print(token, end="", flush=True)
                    resposta_completa += token

            print()  # Quebra de linha ao final da resposta

            # Citação de fontes (páginas consultadas no PDF)
            if "Não tenho informações necessárias" not in resposta_completa and paginas:
                paginas_str = ", ".join(str(p) for p in paginas)
                if USE_RICH:
                    console.print(f"[dim]Fontes consultadas: {settings.PDF_PATH} (Paginas: {paginas_str})[/dim]\n")
                else:
                    print(f"[Fontes consultadas: {settings.PDF_PATH} (Paginas: {paginas_str})]\n")
            else:
                print()

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"\nErro ao processar pergunta: {e}\n")


if __name__ == "__main__":
    main()