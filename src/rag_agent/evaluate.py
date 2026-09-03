"""
El medidor de cobertura: geometría + juicio.

ARQUITECTURA (y por qué es así)

    1. La geometría RECUPERA los 3 pasajes más cercanos. Es barata e instantánea.
    2. Si el mejor pasaje está clarísimamente lejos, es hueco y no gastamos LLM.
    3. En todo lo demás, el LLM LEE y dictamina.

El paso 2 es el ahorro. El paso 3 es el que da la respuesta correcta.

Diseñamos esto DESPUÉS de medir que la geometría sola no alcanzaba: la similitud
coseno mide cercanía TEMÁTICA, no presencia de la respuesta. Ver README.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .ingest import get_vectorstore

# Por debajo de esto ni molestamos al modelo: no hay nada ni remotamente cerca.
# Es un filtro de AHORRO, no el criterio de decisión.
UMBRAL_DESCARTE = 0.35

TOP_K_JUEZ = 3

CUBIERTA = "cubierta"
HUECO = "hueco"
ERROR = "error"  # el juez no produjo veredicto: NO es un hallazgo, es un fallo


@dataclass
class Veredicto:
    pregunta: str
    frecuencia: int
    similitud: float
    estado: str
    fuente: str | None = None
    motivo: str = ""
    uso_llm: bool = False
    docs: list[Document] = field(default_factory=list, repr=False)

    @property
    def es_hueco(self) -> bool:
        return self.estado == HUECO

    @property
    def es_error(self) -> bool:
        return self.estado == ERROR


def recuperar(pregunta: str, k: int = TOP_K_JUEZ) -> list[tuple[Document, float]]:
    """Los k pasajes más cercanos con su similitud (1 - distancia coseno)."""
    store = get_vectorstore()
    return [
        (doc, max(0.0, 1.0 - dist))
        for doc, dist in store.similarity_search_with_score(pregunta, k=k)
    ]


def medir_pregunta(pregunta: str, frecuencia: int = 1, usar_llm: bool = True) -> Veredicto:
    hits = recuperar(pregunta)

    if not hits:
        return Veredicto(pregunta, frecuencia, 0.0, HUECO, motivo="Índice vacío.")

    docs = [d for d, _ in hits]
    mejor_sim = hits[0][1]
    fuente = docs[0].metadata.get("source")

    # Filtro barato: si no hay nada cerca, no hace falta leer.
    if mejor_sim < UMBRAL_DESCARTE:
        return Veredicto(
            pregunta, frecuencia, round(mejor_sim, 3), HUECO, None,
            motivo="Sin pasajes cercanos.", docs=docs,
        )

    if not usar_llm:
        return Veredicto(
            pregunta, frecuencia, round(mejor_sim, 3), CUBIERTA, fuente,
            motivo="(sin juez)", docs=docs,
        )

    from .juez import JuezFallo, juzgar

    try:
        d = juzgar(pregunta, docs)
    except JuezFallo as exc:
        # Un fallo técnico NO se cuenta como hueco. Se reporta aparte.
        return Veredicto(
            pregunta, frecuencia, round(mejor_sim, 3), ERROR, fuente,
            motivo=f"El juez no devolvió veredicto: {exc}", uso_llm=True, docs=docs,
        )

    return Veredicto(
        pregunta=pregunta,
        frecuencia=frecuencia,
        similitud=round(mejor_sim, 3),
        estado=CUBIERTA if d.responde else HUECO,
        fuente=fuente if d.responde else None,
        motivo=d.motivo,
        uso_llm=True,
        docs=docs,
    )


def leer_csv(ruta: Path) -> list[tuple[str, int]]:
    """
    Lee el CSV de preguntas reales.

    Formato mínimo: columna `pregunta`. La columna `frecuencia` es opcional y es
    la que convierte una lista de huecos en una lista PRIORIZADA: no todos los
    huecos duelen igual.
    """
    filas: list[tuple[str, int]] = []
    with ruta.open(encoding="utf-8-sig", newline="") as fh:
        for fila in csv.DictReader(fh):
            texto = (fila.get("pregunta") or "").strip()
            if not texto:
                continue
            try:
                frec = int(fila.get("frecuencia") or 1)
            except ValueError:
                frec = 1
            filas.append((texto, frec))
    return filas


def medir_lote(ruta_csv: Path, usar_llm: bool = True, on_progress=None,
               limite: int = 0) -> list[Veredicto]:
    filas = leer_csv(ruta_csv)
    if limite:
        filas = filas[:limite]
    out: list[Veredicto] = []
    for i, (p, f) in enumerate(filas, start=1):
        if on_progress:
            on_progress(i, len(filas), p)
        out.append(medir_pregunta(p, f, usar_llm=usar_llm))
    return out


def puntaje(veredictos: list[Veredicto]) -> float:
    """
    El número de portada: % de la DEMANDA REAL que está cubierta.

    Ponderado por frecuencia, no por cantidad de preguntas: un hueco que
    preguntan 140 veces pesa más que uno de 16. Esa ponderación es la diferencia
    entre una métrica de vanidad y una lista de trabajo priorizada.

    Las preguntas con ERROR quedan FUERA del cálculo. Contarlas como huecos
    inflaría el problema con fallas nuestras; contarlas como cubiertas lo
    escondería. Se excluyen y se reportan aparte.
    """
    juzgados = [v for v in veredictos if v.estado != ERROR]
    if not juzgados:
        return 0.0
    total = sum(v.frecuencia for v in juzgados)
    ganado = sum(v.frecuencia for v in juzgados if v.estado == CUBIERTA)
    return round(100 * ganado / total, 1)
