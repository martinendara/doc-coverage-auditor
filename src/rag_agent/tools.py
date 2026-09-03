"""
El retrieval expuesto como HERRAMIENTA, no como paso fijo de una cadena.

Esta es la diferencia entre RAG clásico y RAG agéntico, y es lo que hoy
se espera que sepas:

  RAG clásico:  pregunta -> retrieve SIEMPRE -> stuff en el prompt -> respuesta
  RAG agéntico: el modelo DECIDE si buscar, con qué query, y cuántas veces.

Ventaja concreta: "hola" no dispara una búsqueda inútil, y una pregunta
compuesta puede disparar tres búsquedas con queries distintas.
El costo: perdés determinismo. Es un trade-off real, no una mejora gratis.
"""

from __future__ import annotations

from langchain.tools import tool

from .config import settings
from .ingest import get_vectorstore

# El docstring de la función ES el prompt que ve el modelo para decidir
# si usa la herramienta. Escribirlo mal es el bug más común en agentes.


@tool
def buscar_en_documentos(query: str) -> str:
    """Busca pasajes relevantes en la base de documentos del usuario.

    Usá esta herramienta cada vez que la pregunta requiera información
    específica contenida en los documentos indexados. Formulá la query con
    los términos que esperás encontrar en el texto, no repitiendo la
    pregunta del usuario palabra por palabra.

    Args:
        query: términos de búsqueda semántica.

    Returns:
        Pasajes numerados, cada uno con su archivo de origen.
    """
    store = get_vectorstore()
    hits = store.similarity_search(query, k=settings.top_k)

    if not hits:
        return "Sin resultados para esa query en los documentos indexados."

    partes = []
    for i, doc in enumerate(hits, start=1):
        fuente = doc.metadata.get("source", "desconocido")
        partes.append(f"[{i}] fuente: {fuente}\n{doc.page_content}")
    return "\n\n---\n\n".join(partes)


@tool
def listar_fuentes() -> str:
    """Lista los archivos que están indexados y disponibles para consulta.

    Útil cuando el usuario pregunta qué información tenés disponible o si
    un tema está cubierto por los documentos.
    """
    store = get_vectorstore()
    data = store.get(include=["metadatas"])
    fuentes = sorted({m.get("source", "?") for m in data.get("metadatas", [])})
    if not fuentes:
        return "No hay documentos indexados todavía. Corré `python -m rag_agent.ingest`."
    return "Documentos indexados:\n" + "\n".join(f"- {f}" for f in fuentes)


TOOLS = [buscar_en_documentos, listar_fuentes]
