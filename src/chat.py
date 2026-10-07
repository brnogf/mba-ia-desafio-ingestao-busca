import os
import sys

# Garante que a pasta src esteja acessível tanto rodando 'python src/chat.py' quanto 'python chat.py'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from search import search_prompt
except ImportError:
    from src.search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("\n=== Chatbot RAG Full Cycle ===")
    print("Digite sua pergunta ou 'sair' para encerrar.\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
            if not pergunta:
                continue

            if pergunta.lower() in ["sair", "exit", "quit", "q"]:
                print("Encerrando...")
                break

            resposta = chain.invoke(pergunta)
            print(f"RESPOSTA: {resposta}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}\n")


if __name__ == "__main__":
    main()