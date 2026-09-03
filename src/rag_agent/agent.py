"""
El agente. Este archivo es el corazón de LangChain 1.x.

Todo lo que importa está acá y son ~40 líneas útiles:
  1. create_agent(model, tools, system_prompt, middleware)
  2. middleware para controlar el loop
  3. invoke() con una lista de mensajes

Si entendés este archivo entendiste el framework. El resto son integraciones.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_tool_call

from .config import get_chat_model, settings
from .tools import TOOLS

SYSTEM_PROMPT = """Sos un asistente que responde EXCLUSIVAMENTE en base a los \
documentos indexados del usuario.

Reglas:
- Antes de responder cualquier pregunta de contenido, usá `buscar_en_documentos`.
- Citá siempre el archivo de origen entre corchetes, así: [nombre.md].
- Si los pasajes recuperados no contienen la respuesta, decilo de forma \
explícita: "Eso no está en los documentos". No completes con conocimiento general.
- Si una pregunta tiene varias partes, hacé varias búsquedas con queries distintas.
- Respondé en el idioma en que te hablen."""


@wrap_tool_call
def log_tool_calls(request, handler):
    """
    Middleware propio: envuelve CADA llamada a herramienta.

    `wrap_tool_call` te da el request antes y el resultado después, así que
    acá podrías cachear, reintentar, censurar o medir latencia. Es el hook
    más útil para debuggear por qué un agente hace lo que hace.
    """
    nombre = request.tool_call["name"]
    args = request.tool_call.get("args", {})
    print(f"  ↳ tool: {nombre}({args})")
    return handler(request)


def build_agent(verbose: bool = True):
    """Construye el agente RAG. Devuelve un grafo compilado de LangGraph."""
    middleware = [
        # Techo duro al loop. Con modelos locales chicos esto no es opcional:
        # un 8B puede entrar en bucle llamando la misma tool 20 veces.
        ToolCallLimitMiddleware(thread_limit=8),
    ]
    if verbose:
        middleware.append(log_tool_calls)

    return create_agent(
        model=get_chat_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
    )


def ask(question: str, verbose: bool = True) -> str:
    """Atajo de una sola pregunta, sin historial."""
    agent = build_agent(verbose=verbose)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys

    print(f"[{settings.llm_provider}] {ask(' '.join(sys.argv[1:]))}")
