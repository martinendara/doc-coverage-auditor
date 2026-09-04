"""
El juez: decide si los pasajes recuperados REALMENTE responden la pregunta.

POR QUÉ EXISTE
La similitud coseno mide "¿habla del mismo tema?", no "¿contiene la respuesta?".
Lo medimos y falló: "¿Se integra con Salesforce?" sacó 0.77 contra un documento
de integraciones que no menciona Salesforce. La geometría trae candidatos; el
juicio lo tiene que hacer un modelo que LEA.

POR QUÉ NO USAMOS with_structured_output CON MODELOS LOCALES
LangChain ofrece `with_structured_output`, que obliga al modelo a devolver un
objeto. Es la forma correcta con proveedores cloud. Contra LM Studio fallaron
los tres métodos, cada uno por una razón distinta:

    json_mode         -> LM Studio solo acepta 'json_schema' o 'text'
    json_schema       -> devuelve sin el campo 'parsed' que LangChain espera
    function_calling  -> no soporta forzar una herramienta puntual

"Compatible con OpenAI" es compatible hasta cierto punto. Cuando la abstracción
se filtra, se baja un nivel: pedimos JSON por prompt, recibimos texto, lo
limpiamos y lo validamos con Pydantic. Es más código pero funciona en cualquier
backend, y es el patrón que se usa en producción justamente por eso.

Con proveedores cloud sí se puede usar la vía elegante: JUEZ_MODO=structured.
"""

from __future__ import annotations

import json
import os
import re

from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError

from .config import get_chat_model


class Dictamen(BaseModel):
    """La forma exacta que esperamos de vuelta."""

    responde: bool = Field(
        description="true si los pasajes contienen la información pedida"
    )
    motivo: str = Field(description="Una frase breve justificando el veredicto")


class JuezFallo(Exception):
    """
    El modelo no produjo un veredicto utilizable.

    Existe como excepción propia y NO como un Dictamen(responde=False) por una
    razón de honestidad del producto: un fallo técnico no puede disfrazarse de
    hallazgo. Si contáramos los fallos como huecos, le estaríamos reportando al
    usuario problemas de documentación que en realidad son problemas nuestros.
    """ 


PROMPT = """Sos un auditor de documentación. Tu tarea NO es responder la pregunta.
Tu tarea es decidir si los pasajes que te doy contienen la información pedida.

Criterio para responder true: un usuario que lea estos pasajes obtendría
información sustantiva sobre lo que pregunta. NO hace falta que haya un
instructivo paso a paso; si el texto explica las condiciones, las reglas o los
datos concretos del tema, alcanza.

Criterio para responder false: el tema puntual de la pregunta no aparece.
Que un pasaje hable del MISMO ÁREA no significa que responda: si preguntan por
reembolsos y el texto habla de facturación pero nunca menciona reembolsos, es
false. No completes con conocimiento general ni con lo que suele ser cierto en
otras empresas. Solo con lo que está escrito.

PREGUNTA:
{pregunta}

PASAJES:
{pasajes}

Respondé ÚNICAMENTE con un objeto JSON, sin texto antes ni después:
{{"responde": true, "motivo": "una frase breve en español"}}"""


def _formatear(docs: list[Document]) -> str:
    partes = []
    for i, d in enumerate(docs, start=1):
        fuente = d.metadata.get("source", "?")
        partes.append(f"--- pasaje {i} (de {fuente}) ---\n{d.page_content}")
    return "\n\n".join(partes) if partes else "(no se recuperó ningún pasaje)"


def extraer_json(texto: str) -> dict:
    """
    Saca el objeto JSON de una respuesta que puede venir sucia.

    Los modelos de razonamiento envuelven su pensamiento en <think>...</think>.
    Muchos modelos rodean el JSON con ```json ... ```. Y varios agregan una
    frase de cortesía antes o después. Limpiamos las tres cosas y nos quedamos
    con el primer objeto balanceado que encontremos.
    """
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)

    # Bloque de pensamiento SIN cerrar: el modelo se quedó sin tokens en medio
    # del razonamiento. Lo que sigue no es respuesta, es pensamiento cortado.
    if "<think>" in texto:
        raise ValueError("El modelo se quedó sin tokens razonando (<think> sin cerrar).")

    texto = re.sub(r"```(?:json)?", "", texto).strip()

    inicio = texto.find("{")
    if inicio == -1:
        raise ValueError(f"Sin JSON en la respuesta: {texto[:200]!r}")

    profundidad = 0
    for i, ch in enumerate(texto[inicio:], start=inicio):
        if ch == "{":
            profundidad += 1
        elif ch == "}":
            profundidad -= 1
            if profundidad == 0:
                return json.loads(texto[inicio : i + 1])

    raise ValueError(f"JSON incompleto: {texto[:200]!r}")


def _contenido(msg) -> str:
    """
    Saca el texto de la respuesta.

    Los modelos de razonamiento a veces devuelven `content` vacío y dejan todo
    en un campo aparte de razonamiento. Miramos ahí también antes de rendirnos.
    """
    if msg.content:
        return msg.content
    extra = getattr(msg, "additional_kwargs", {}) or {}
    for clave in ("reasoning_content", "reasoning"):
        if extra.get(clave):
            return str(extra[clave])
    return ""


def juzgar(pregunta: str, docs: list[Document]) -> Dictamen:
    """
    Una llamada al LLM por pregunta. Local y gratis, o cloud según el .env.

    Levanta JuezFallo si el modelo no produce un veredicto utilizable.
    """
    if not docs:
        return Dictamen(responde=False, motivo="No se recuperó ningún pasaje.")

    texto_prompt = PROMPT.format(pregunta=pregunta, pasajes=_formatear(docs))
    max_tokens = int(os.getenv("JUEZ_MAX_TOKENS", "800"))

    # Vía elegante, solo para proveedores que la soportan de verdad.
    if os.getenv("JUEZ_MODO", "manual") == "structured":
        metodo = os.getenv("STRUCTURED_OUTPUT_METHOD", "json_schema")
        modelo = get_chat_model(temperature=0).with_structured_output(
            Dictamen, method=metodo
        )
        return modelo.invoke(texto_prompt)

    # Vía manual: funciona contra cualquier backend.
    debug = os.getenv("JUEZ_DEBUG") == "1"

    # El modelo razona antes de responder y a veces agota el presupuesto
    # antes de escribir el JSON: `content` vuelve vacío. Medido: 138 tokens
    # de razonamiento para un veredicto de 18. El reintento sube el techo,
    # no cambia la redacción — la redacción ya se probó y no era la causa.
    intentos = [
        (texto_prompt, max_tokens),
        (texto_prompt, max_tokens * 2),
        (texto_prompt + "\n\nRespondé AHORA con una sola línea JSON. Nada más.",
         max_tokens * 4),
    ]
    ultimo_error = None
    for i, (prompt, techo) in enumerate(intentos):
        modelo = get_chat_model(temperature=0, max_tokens=techo)
        crudo = _contenido(modelo.invoke(prompt))
        if debug:
            print(f"\n--- intento {i + 1}, respuesta cruda ---\n{crudo!r}\n")
        try:
            return Dictamen(**extraer_json(crudo))
        except (ValueError, ValidationError, json.JSONDecodeError, TypeError) as exc:
            ultimo_error = exc

    raise JuezFallo(str(ultimo_error)[:160])
