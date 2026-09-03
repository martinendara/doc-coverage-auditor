# Estado del proyecto

Última actualización: 3 de septiembre de 2026.
Documento vivo: se actualiza al cerrar cada sesión, no se archiva.

---

## Dónde estamos

El pipeline completo funciona de punta a punta. Primer commit subido a
`github.com/martinendara/doc-coverage-auditor` (público, MIT).

Última corrida completa sobre el corpus de ejemplo:

```
49.3% de cobertura · 20 preguntas · 1 sin veredicto
9/9 huecos plantados detectados · 10/10 cubiertas correctas · 1 FALLO
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

`JUEZ_MAX_TOKENS=2500` **no es negociable con Qwen3.5**. Con 800 el modelo se
queda sin presupuesto razonando y devuelve vacío: rompió los 20 veredictos de
una corrida entera, en silencio.

**Ritual de cada sesión:**

```bash
cd <carpeta del proyecto>
source .venv/bin/activate
export PYTHONPATH=src
```

---

## Pendiente 1 — el FALLO que queda

Una pregunta de veinte no obtiene veredicto: el modelo devuelve `content` vacío
después de razonar. Aparece como `FALLO` y queda excluida del puntaje, así que
no corrompe el resultado, pero hay que cerrarlo.

Ya probado y descartado:
- Tres intentos con prompts progresivamente más explícitos (incluido `/no_think`)
- Leer el campo de razonamiento cuando `content` viene vacío
- Bajar `max_tokens` (empeoró todo)

Caminos abiertos, en orden:
1. **Apagar el razonamiento por API.** LM Studio no tiene interruptor en la UI
   del servidor (el toggle de Bionic es solo para su ventana de chat, no afecta
   a la API). Requiere pasar un parámetro extra en la llamada — cambio de código
   en `config.py`.
2. **Modelo sin razonamiento.** Gemma 4 E4B o Ministral 3 3B. Más rápidos,
   no divagan, y la tarea no necesita cadena de pensamiento. Probablemente la
   opción más simple y la que además resuelve el problema de velocidad.

---

## Pendiente 2 — el front

Objetivo de producto: arrastrar una carpeta, ver el número grande, la lista de
huecos priorizada, y un chat abajo sobre los mismos documentos.

Decidido: **FastAPI + HTML simple**, no Streamlit. Streamlit se resuelve en una
tarde pero visualmente grita "demo"; el punto del ejercicio es que parezca
producto terminado.

Restricción medida que hay que resolver: el juicio es **secuencial**, una
llamada por pregunta. Veinte preguntas contra el 9B local tardaron varios
minutos y calentaron la máquina. Con 200 son horas. Tres caminos:
cachear veredictos por (pregunta, hash del chunk), usar un modelo más chico solo
para juzgar, o paralelizar. Decidir con el tiempo medido, no antes.

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

---

## Trampas que ya costaron tiempo

1. **Chroma usa L2 por defecto.** Sin `hnsw:space=cosine` la similitud no se
   puede interpretar ni comparar contra un umbral.
2. **Nomic necesita prefijos de tarea.** Sin ellos degrada en silencio.
   Cambiar los prefijos obliga a `rm -rf .chroma` y reindexar: los vectores
   viejos no son comparables con los nuevos.
3. **La carpeta del proyecto está anidada** dentro de otra del mismo nombre.
   Verificar con `ls` que se ve `requirements.txt` antes de correr nada.
4. **Correr los tests antes de cada commit.** Una vez el archivo de tests quedó
   en `src/` en vez de `tests/` y el repo casi sube con 3 tests en vez de 9.

---

## Después del front

- Postear en LinkedIn (borradores listos; esperar a tener front y más commits)
- Agregar badges de LangChain/LangGraph/Pydantic/pytest al perfil de GitHub
- Extensión de mayor valor pendiente: un set de evaluación propio del juez
  (preguntas con veredicto esperado) para medir el prompt en vez de estimarlo
