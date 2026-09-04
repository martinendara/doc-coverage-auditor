# Estado del proyecto

Última actualización: 4 de septiembre de 2026.
Documento vivo: se actualiza al cerrar cada sesión, no se archiva.

---

## Dónde estamos

El pipeline completo funciona de punta a punta. Repo público en
`github.com/martinendara/doc-coverage-auditor` (MIT).

Última corrida completa sobre el corpus de ejemplo:

```
50.2% de cobertura · 20 preguntas · 20 con veredicto
9/9 huecos plantados detectados · 11/11 cubiertas correctas · 0 FALLOS
Tiempo de corrida local: 44:47 (secuencial, 9B en LM Studio)
```

Tests: 9 passed, sin API keys ni servidor local.

---

## Configuración que funciona (no cambiar sin medir)

**LM Studio** con dos modelos cargados a la vez:

| Rol | Identificador exacto |
|---|---|
| Embeddings | `text-embedding-nomic-embed-text-v1.5` |
| Chat / juez | `qwen/qwen3.5-9b` |

```bash
~/.lmstudio/bin/lms load text-embedding-nomic-embed-text-v1.5
~/.lmstudio/bin/lms load qwen/qwen3.5-9b
~/.lmstudio/bin/lms server start
curl http://localhost:1234/v1/models   # verificar
```

**Valores calibrados en `.env`:**

```
LOCAL_CHAT_MODEL=qwen/qwen3.5-9b
LOCAL_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
EMBED_DOC_PREFIX=search_document:
EMBED_QUERY_PREFIX=search_query:
JUEZ_MAX_TOKENS=2500
CHUNK_SIZE=400
CHUNK_OVERLAP=80
```

`JUEZ_MAX_TOKENS=2500` es el **techo del primer intento**, no un mínimo. Desde
el fix del 4 de septiembre el juez escala solo (×1, ×2, ×4), así que el valor
dejó de ser crítico. Bajarlo es un experimento abierto: podría acelerar las
preguntas fáciles y dejar que las difíciles se recuperen en el segundo intento.

Los prefijos de Nomic quedan **sin espacio final** y así están medidos.
Verificado que `dotenv` los lee simétricos (`'search_document:'` /
`'search_query:'`), así que documentos y consultas se embeben igual. Agregarles
el espacio canónico es un experimento pendiente que obliga a reindexar.

**Modelos cloud (bloque comentado en `.env`, identificadores al 4/9/2026):**

```
# ANTHROPIC_MODEL=claude-sonnet-5
# OPENAI_MODEL=gpt-5.5
# GEMINI_MODEL=gemini-3.8-flash
# DEEPSEEK_MODEL=deepseek-v4-pro
```

Notas: Gemini 3.8 rechaza `temperature`, `top_p` y `top_k`, y no soporta
`thinking_level: minimal` — no sirve como juez sin razonamiento. DeepSeek sí
acepta `thinking: {"type": "disabled"}` y `deepseek-v4-flash` cuesta centavos,
así que es el mejor candidato a juez cloud barato.

**Ritual de cada sesión:**

```bash
cd <carpeta del proyecto>
source .venv/bin/activate
export PYTHONPATH=src
```

---

## Cerrado — el FALLO (4 de septiembre)

Las 20 preguntas obtienen veredicto. La que fallaba ahora resuelve, y resuelve
bien: entró como cubierta. La cobertura subió de 49.3% a 50.2% solo porque una
pregunta que quedaba excluida del puntaje volvió a contar.

**Causa real.** El reintento variaba la **redacción** del prompt, pero el modelo
se construía una sola vez **fuera** del loop, con `max_tokens` fijo. Los tres
intentos corrían con el mismo techo, así que los tres se cortaban en el mismo
lugar. Era presupuesto, no redacción. Reintentar con otras palabras era como
repetir una llamada que se corta por señal hablando más claro.

**Fix.** En `juez.py`, `intentos` pasó a ser una lista de tuplas
`(prompt, techo)` con techos ×1 / ×2 / ×4, y el modelo se reconstruye **dentro**
del loop con `max_tokens=techo`.

**Descartado con evidencia — no reabrir:**

- **Apagar el razonamiento por API.** `chat_template_kwargs: {enable_thinking:
  false}` es aceptado por LM Studio y **ignorado** por el modelo. Medido con
  curl: 138 tokens de razonamiento para un veredicto de 18 tokens de contenido.
  `/no_think` tampoco funciona: es una instrucción dentro del texto, y el modelo
  la puede ignorar.
- **Modelo sin razonamiento.** Se bajó Gemma 4 E4B y **razonó el doble que
  Qwen** (286 tokens contra 138) para la misma respuesta trivial, pesando más
  (6.86 GB contra 5.98) y siendo peor juez (4B efectivo contra 9B). La premisa
  "más rápidos, no divagan" era de otra época. Borrado del disco. No bajar
  Ministral 3B esperando otro resultado.

---

## Pendiente 1 — el front

Objetivo de producto: arrastrar una carpeta, ver el número grande, la lista de
huecos priorizada, y un chat abajo sobre los mismos documentos.

Decidido: **FastAPI + HTML simple**, no Streamlit. Streamlit se resuelve en una
tarde pero visualmente grita "demo"; el punto del ejercicio es que parezca
producto terminado.

**Sobre el tiempo de corrida.** Los 44:47 son límite de RAM corriendo un 9B en
local, no un defecto del código (`1.60s user · 0% cpu`: el proceso Python espera
al servidor, no trabaja). La vía cloud ya está resuelta por variable de entorno
y baja esto a segundos. **El front va contra cloud**; local queda como modo de
aprendizaje. Las tres optimizaciones anotadas antes (caché por pregunta + hash
del chunk, modelo chico solo para juzgar, paralelizar) quedan disponibles pero
no son bloqueantes.

En web el navegador **sube copias** de los archivos, no da acceso a la carpeta.
Solo una app local puede apuntar a una ruta real. Definir cuál de las dos es.

---

## Decisiones tomadas (no rediscutir sin motivo nuevo)

- **Geometría recupera, LLM juzga.** La similitud coseno sola dio 0/9. Está
  medido y documentado en el README.
- **Los fallos técnicos no se cuentan como huecos.** Estado `ERROR` aparte,
  excluido del puntaje.
- **El puntaje se pondera por frecuencia**, no por cantidad de preguntas.
- **Parser manual de JSON en vez de `with_structured_output`** para backends
  locales. Los tres métodos de LangChain fallaron contra LM Studio, cada uno
  distinto. `JUEZ_MODO=structured` recupera la vía elegante en cloud.
- **Local por defecto.** Cloud es un cambio de variable de entorno, pero el
  objetivo de aprendizaje es local.
- **Auditar antes de escribir código.** El fix del FALLO salió de tres curls,
  un `grep` y leer 50 líneas. Ninguna de las hipótesis del handover anterior
  sobrevivió a la medición.

---

## Trampas que ya costaron tiempo

1. **Chroma usa L2 por defecto.** Sin `hnsw:space=cosine` la similitud no se
   puede interpretar ni comparar contra un umbral.
2. **Nomic necesita prefijos de tarea.** Sin ellos degrada en silencio.
   Cambiar los prefijos obliga a `rm -rf .chroma` y reindexar: los vectores
   viejos no son comparables con los nuevos.
3. **La carpeta del proyecto está anidada** dentro de otra del mismo nombre.
   La buena es la que tiene `requirements.txt`. Verificar con `ls` antes de
   correr nada.
4. **Correr los tests antes de cada commit.** Una vez el archivo de tests quedó
   en `src/` en vez de `tests/` y el repo casi sube con 3 tests en vez de 9.
5. **Los modelos de 2026 razonan de fábrica.** No hay interruptor confiable por
   API, y los modelos más chicos no razonan menos — a veces razonan más. Si
   `content` vuelve vacío, la causa es presupuesto: subí el techo de tokens, no
   reescribas el prompt.
6. **Los defaults del código mienten.** `config.py:36` cae a `qwen/qwen3-8b`
   (un modelo que no está bajado) y `juez.py:151` cae a `800` tokens (el valor
   que rompía los veredictos). Quien clone el repo sin `.env` arranca en la
   configuración que ya sabemos que falla. Pendiente corregirlos.
7. **Los tests pasan sin servidor local.** 9 passed no verifica el juez. El
   único chequeo real es la corrida completa del reporte.

---

## Después del front

- Corregir los defaults de `config.py` y `juez.py` (trampa 6)
- Experimento: bajar `JUEZ_MAX_TOKENS` a 1200 y medir si el escalado compensa
- Postear en LinkedIn (borradores listos; esperar a tener front y más commits)
- Agregar badges de LangChain/LangGraph/Pydantic/pytest al perfil de GitHub
- Extensión de mayor valor pendiente: un set de evaluación propio del juez
  (preguntas con veredicto esperado) para medir el prompt en vez de estimarlo
