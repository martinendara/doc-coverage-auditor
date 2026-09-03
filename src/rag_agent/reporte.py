"""
Reporte en terminal. Es el producto sin cara todavía.

    python -m rag_agent.reporte data/preguntas.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import CUBIERTA, ERROR, HUECO, medir_lote, puntaje

ETIQUETA = {CUBIERTA: "OK   ", HUECO: "HUECO", ERROR: "FALLO"}


def _progreso(i: int, total: int, pregunta: str) -> None:
    print(f"\r  analizando {i}/{total}...", end="", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medir cobertura de la documentación")
    parser.add_argument("csv", type=Path, nargs="?", default=Path("data/preguntas.csv"))
    parser.add_argument("--sin-juez", action="store_true",
                        help="Solo geometría, sin LLM (rápido, poco confiable)")
    parser.add_argument("--limite", type=int, default=0,
                        help="Analizar solo las primeras N preguntas (para probar rápido)")
    args = parser.parse_args()

    veredictos = medir_lote(args.csv, usar_llm=not args.sin_juez,
                            on_progress=_progreso, limite=args.limite)
    print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    if not veredictos:
        raise SystemExit(f"No hay preguntas en {args.csv}")

    print()
    print(f"  Tu documentación responde el {puntaje(veredictos)}%")
    print(f"  de lo que te preguntan ({len(veredictos)} preguntas analizadas)")
    print()

    errores = [v for v in veredictos if v.es_error]
    if errores:
        print(f"  ({len(errores)} preguntas sin veredicto, excluidas del cálculo)")
        print()

    # Huecos primero, y dentro de cada grupo por frecuencia: dolor real.
    orden = sorted(veredictos, key=lambda v: (not v.es_hueco, v.es_error, -v.frecuencia))

    for v in orden:
        print(f"  {ETIQUETA[v.estado]} {v.similitud:.2f}  {v.frecuencia:>4}x  {v.pregunta}")
        detalle = v.fuente if v.estado == CUBIERTA else v.motivo
        print(f"{'':21}└─ {detalle}")

    huecos = [v for v in veredictos if v.es_hueco]
    if huecos:
        perdidas = sum(v.frecuencia for v in huecos)
        print()
        print(f"  {len(huecos)} huecos = {perdidas} consultas sin respuesta.")
    print()


if __name__ == "__main__":
    main()
