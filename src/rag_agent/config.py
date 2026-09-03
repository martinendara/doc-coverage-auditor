"""
Configuración y fábrica de modelos.

La idea central: el resto del código NUNCA sabe contra qué proveedor corre.
Pide `get_chat_model()` o `get_embeddings()` y listo. Cambiar de LM Studio
local a Claude es cambiar una variable de entorno, no tocar código.

Esa indirección es, básicamente, el 80% del valor real de LangChain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = Path(os.getenv("DOCS_DIR", PROJECT_ROOT / "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / ".chroma"))
COLLECTION = os.getenv("COLLECTION_NAME", "docs")


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "local")
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "local")

    # LM Studio expone una API compatible con OpenAI en /v1
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1")
    local_chat_model: str = os.getenv("LOCAL_CHAT_MODEL", "qwen/qwen3-8b")
    local_embed_model: str = os.getenv(
        "LOCAL_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"
    )

    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))


settings = Settings()


def get_chat_model(**kwargs) -> BaseChatModel:
    """Devuelve el modelo de chat según LLM_PROVIDER."""
    provider = settings.llm_provider.lower()

    if provider == "local":
        from langchain_openai import ChatOpenAI

        # LM Studio ignora la api_key pero el cliente exige que exista.
        return ChatOpenAI(
            model=settings.local_chat_model,
            base_url=settings.local_base_url,
            api_key=os.getenv("LOCAL_API_KEY", "lm-studio"),
            temperature=kwargs.pop("temperature", 0.1),
            **kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=kwargs.pop("temperature", 0.1),
            **kwargs,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=kwargs.pop("temperature", 0.1),
            **kwargs,
        )

    raise ValueError(
        f"LLM_PROVIDER desconocido: {provider!r}. Usá local | anthropic | openai."
    )


class PrefixedEmbeddings(Embeddings):
    """
    Envuelve un modelo de embeddings para anteponer prefijos de tarea.

    Algunos modelos (nomic-embed, e5, bge) NO fueron entrenados para tratar
    igual a un documento y a una pregunta. Esperan que les avises cuál es
    cuál mediante un prefijo en el texto:

        search_document: <el contenido>
        search_query: <la pregunta>

    Sin eso el modelo colapsa todas las distancias hacia el centro y la
    similitud pierde poder de discriminación. Es un detalle que no aparece
    en ningún tutorial y arruina silenciosamente cualquier medición.
    """

    def __init__(self, inner: Embeddings, doc_prefix: str, query_prefix: str):
        self.inner = inner
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


def get_embeddings() -> Embeddings:
    """
    Devuelve el modelo de embeddings según EMBEDDINGS_PROVIDER.

    OJO: Anthropic no ofrece API de embeddings. Si usás LLM_PROVIDER=anthropic
    igual necesitás EMBEDDINGS_PROVIDER=local u openai. Son dos ejes distintos
    y por eso están separados.
    """
    provider = settings.embeddings_provider.lower()

    if provider == "local":
        from langchain_openai import OpenAIEmbeddings

        base = OpenAIEmbeddings(
            model=settings.local_embed_model,
            base_url=settings.local_base_url,
            api_key=os.getenv("LOCAL_API_KEY", "lm-studio"),
            check_embedding_ctx_length=False,  # necesario para backends no-OpenAI
        )
        doc_p = os.getenv("EMBED_DOC_PREFIX", "")
        query_p = os.getenv("EMBED_QUERY_PREFIX", "")
        if doc_p or query_p:
            return PrefixedEmbeddings(base, doc_p, query_p)
        return base

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embed_model)

    raise ValueError(
        f"EMBEDDINGS_PROVIDER desconocido: {provider!r}. Usá local | openai."
    )


def describe() -> str:
    return (
        f"LLM: {settings.llm_provider} | "
        f"Embeddings: {settings.embeddings_provider} | "
        f"top_k={settings.top_k}"
    )
