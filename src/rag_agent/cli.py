"""
CLI interactiva con historial.

Detalle que suele confundir: el agente NO tiene memoria propia entre
invocaciones. Vos le pasás la lista completa de mensajes cada vez.
(Existe `checkpointer` en create_agent para persistir estado, pero para
una sesión de terminal esto es más simple y más explícito.)
"""

from __future__ import annotations

import sys

from .agent import build_agent
from .config import describe


def main() -> None:
    print("Agente RAG sobre tus documentos.")
    print(describe())
    print("Escribí tu pregunta. 'salir' para terminar.\n")

    agent = build_agent(verbose=True)
    history: list[dict] = []

    while True:
        try:
            pregunta = input("vos > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pregunta:
            continue
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break

        history.append({"role": "user", "content": pregunta})
        try:
            result = agent.invoke({"messages": history})
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}\n", file=sys.stderr)
            history.pop()
            continue

        history = result["messages"]
        print(f"\nagente > {history[-1].content}\n")


if __name__ == "__main__":
    main()
