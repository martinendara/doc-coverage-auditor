"""
Ingesta: leer archivos -> partirlos en chunks -> embeddings -> Chroma.

Este archivo es la mitad "aburrida" de RAG y es donde se gana o se pierde
la calidad. El agente no puede citar lo que la ingesta nunca indexó.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import COLLECTION, DOCS_DIR, INDEX_DIR, get_embeddings, settings

SUPPORTED = {".md", ".txt", ".markdown"}


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Lee los archivos de texto de docs/ y los envuelve en Document."""
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path.relative_to(docs_dir))},
            )
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    RecursiveCharacterTextSplitter intenta cortar primero por párrafos,
    después por oraciones, y solo al final por caracteres sueltos.
    Es el default sensato: preserva unidades semánticas.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def get_vectorstore() -> Chroma:
    """
    Abre (o crea) el índice persistido en disco.

    `hnsw:space=cosine` NO es un detalle menor: por defecto Chroma usa distancia
    euclidiana (L2), cuyo valor no tiene techo y no se puede interpretar como
    "qué tan parecido es esto". Con coseno la distancia va de 0 a 2, y
    `similitud = 1 - distancia` da un número entre 0 y 1 que sí se puede
    comparar entre preguntas y contra un umbral.

    Todo el medidor de cobertura depende de esto.
    """
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(INDEX_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _stable_id(chunk: Document) -> str:
    """
    ID determinístico por contenido: reindexar dos veces no duplica chunks.
    Detalle chico, pero es la diferencia entre un demo y algo usable.
    """
    raw = f"{chunk.metadata.get('source')}::{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def ingest(docs_dir: Path = DOCS_DIR) -> int:
    docs = load_documents(docs_dir)
    if not docs:
        raise SystemExit(f"No encontré archivos .md/.txt en {docs_dir}")

    chunks = split_documents(docs)
    store = get_vectorstore()
    store.add_documents(chunks, ids=[_stable_id(c) for c in chunks])

    print(f"{len(docs)} archivos -> {len(chunks)} chunks indexados en {INDEX_DIR}")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexar documentos para el agente RAG")
    parser.add_argument("--docs", type=Path, default=DOCS_DIR)
    args = parser.parse_args()
    ingest(args.docs)


if __name__ == "__main__":
    main()
