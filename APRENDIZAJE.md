# Ruta de estudio

Notas para vos, no para el reclutador. El README es la cara pública; esto es el andamio.

## Orden de lectura del repo

Leelo en este orden, no alfabéticamente:

1. **`agent.py`** — Empezá acá. Son 40 líneas útiles y contienen todo LangChain 1.x.
   Si entendés `create_agent(model, tools, system_prompt, middleware)` y el
   `.invoke({"messages": [...]})`, ya sabés el framework. En serio.
2. **`tools.py`** — El decorador `@tool`. Fijate que el docstring no es
   documentación: es el prompt que lee el modelo para decidir si la usa.
3. **`config.py`** — La abstracción de proveedor. Este es el valor real del
   framework: cambiar Claude por un Qwen local sin tocar lógica.
4. **`ingest.py`** — Chunking y embeddings. Nada de esto es LangChain-específico
   conceptualmente, pero las clases sí lo son.
5. **`cli.py`** — Que el agente no tiene memoria propia. Vos pasás el historial.

## Lo que hay que poder explicar en una entrevista

- **Por qué `create_agent` y no `create_react_agent`**: el primero es el estándar
  de 1.0, construido sobre LangGraph, con middleware como punto de extensión.
  El de `langgraph.prebuilt` es la versión anterior, más rígida.
- **Qué es middleware**: hooks que corren dentro del loop del agente
  (`before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call`). Sirven
  para prompts dinámicos, resumen de contexto, límites, guardrails, PII.
- **RAG agéntico vs RAG clásico**: el trade-off de determinismo. Tenelo listo,
  es la pregunta que separa a quien leyó un tutorial de quien construyó algo.
- **Por qué LLM y embeddings son ejes separados**: Anthropic no tiene embeddings.
- **Qué es LangGraph y cómo se relaciona**: `create_agent` devuelve un grafo
  compilado de LangGraph. LangChain es la capa de conveniencia; LangGraph es el
  runtime de abajo, para topologías que no son "loop hasta terminar".

## Los tres puntos donde se rompe en la práctica

1. **El modelo no llama la tool.** Casi siempre es el docstring. Segundo
   sospechoso: el modelo local no soporta tool calling bien.
2. **Loop infinito.** De ahí `ToolCallLimitMiddleware`. Con modelos chicos es
   obligatorio, no opcional.
3. **Retrieval trae basura.** No es problema del agente, es de la ingesta:
   chunk_size mal calibrado o separadores que parten a mitad de idea.

## Extensiones (en orden de rendimiento por hora invertida)

1. **Reranking** — Recuperá `k=20`, pasalo por un cross-encoder, quedate con 4.
   Es la mejora de calidad más grande por línea de código en todo RAG.
2. **`response_format`** — `create_agent` acepta un modelo Pydantic para salida
   estructurada. Convierte al agente en algo que podés meter en un pipeline.
3. **Evaluación** — 20 preguntas con respuesta esperada y un script que mida
   aciertos. Esto es lo que más te va a diferenciar: casi nadie lo hace en un
   repo de portfolio, y es exactamente lo que se hace en producción.
4. **`checkpointer`** — Persistir conversaciones entre sesiones.
5. **Un segundo loader** (PDF) — El más obvio y el de menor aprendizaje.

Si vas a hacer solo una, hacé la 3.

## Advertencia honesta sobre el modelo local

Qwen3-8B en 4-bit hace tool calling, pero de forma inconsistente: a veces
inventa el nombre de la tool, a veces devuelve JSON mal formado. Para *aprender
el framework* alcanza y te sale gratis. Para *demostrar que el proyecto anda*
en un video o una demo, corré una vez contra Claude y grabá eso. No es hacer
trampa: es separar la limitación del modelo de la corrección de tu código.
