# Documentation Coverage Auditor

Point it at a folder of documents and a CSV of questions your users actually ask.
It tells you **what percentage of real demand your documentation answers**, and
lists the gaps ordered by how often people hit them.

Built on **LangChain 1.x**. Runs against a **local model via LM Studio** or a
**cloud provider** — one environment variable switches between them.

Real output on the bundled sample corpus (documents and questions are in Spanish):

```
  Tu documentación responde el 49.3%
  de lo que te preguntan (20 preguntas analizadas)

  (1 preguntas sin veredicto, excluidas del cálculo)

  HUECO 0.77    38x  ¿Se integra con Salesforce?
                     └─ Los pasajes no mencionan la integración con Salesforce.
  HUECO 0.63   138x  ¿Cómo pido un reembolso?
                     └─ Los pasajes no mencionan el proceso de reembolso.
  OK    0.67   142x  ¿Cómo cambio de plan?
                     └─ facturacion.md
  ...
  FALLO 0.71    24x  ¿Cómo reporto un problema urgente?
                     └─ El juez no devolvió veredicto.

  9 huecos = 615 consultas sin respuesta.
```

---

## The finding this project is built on

The obvious way to detect a gap is geometry: embed the question, find the nearest
passage, and if it's far away there's no answer. It costs nothing and runs in
milliseconds.

**It doesn't work.** Measured on a 4-document corpus with 9 deliberately planted
gaps, every question — answerable or not — scored between 0.60 and 0.77 cosine
similarity. No threshold separates them.

The clearest case: *"Do you integrate with Salesforce?"* scored **0.77, the
second-highest similarity in the entire set** — against an integrations document
that lists Slack, Google Workspace and Teams, and never mentions Salesforce.

**Cosine similarity measures topical proximity, not answer presence.** In a
single-domain corpus everything is topically close to everything, so the signal
saturates. Retrieval is a good *candidate generator* and a bad *judge*.

So the architecture splits the two jobs:

| Stage | Does what | Cost |
|---|---|---|
| Geometry | Retrieves the 3 nearest passages, discards the obviously distant | ~0 |
| LLM judge | Reads them and rules on whether the answer is present | 1 call per question, once |

With the judge in place: **19 of 19 questions that received a verdict were
correct** — all 9 planted gaps flagged, all 10 answerable questions passed,
including the 0.77 Salesforce case. The 20th returned no usable verdict and is
reported as `ERROR`, excluded from the score rather than counted as a gap.

---

### Measured accuracy

The sample corpus has 9 gaps planted deliberately (refunds, account cancellation,
data export, mobile app, SSO/SAML, GDPR, Salesforce, non-profit discounts, SLA)
against 11 questions the documents genuinely answer.

| | Result |
|---|---|
| Planted gaps detected | **9 / 9** |
| Answerable questions correctly passed | **10 / 10** |
| Technical failures | 1, excluded from the score, not counted as a gap |

Geometry alone scored **0 / 9** on the same corpus: with no judge, every question
landed above any usable threshold and the tool reported 100% coverage.

---

## Architecture

```
docs/ ──▶ ingest.py ──▶ Chroma (cosine)
                            │
questions.csv ──▶ evaluate.py ──┴──▶ juez.py ──▶ report
                  (retrieves)        (rules)
```

| Module | Responsibility |
|---|---|
| `config.py` | Provider-agnostic model factory (local / Anthropic / OpenAI) |
| `ingest.py` | Chunking, embeddings, persistence |
| `evaluate.py` | Retrieval, thresholds, scoring |
| `juez.py` | The LLM judge and its tolerant JSON parser |
| `tools.py` | Retrieval exposed as a `@tool` for the agent |
| `agent.py` | `create_agent` + middleware — chat over the same corpus |
| `reporte.py` | Terminal report |

Verified against `langchain==1.3.18`.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | **LangChain 1.3** | Current API: `create_agent`, middleware, `@tool`, `with_structured_output`. The deprecated chain APIs (`LLMChain`, `SequentialChain`, `ConversationBufferMemory`) now live in `langchain-classic` and are **not** used here. |
| Agent runtime | **LangGraph 1.2** | `create_agent` compiles to a LangGraph graph; middleware hooks into that loop. |
| Vector store | **Chroma 1.5** | Embedded, persists to disk, no server to run. Forced to cosine distance so similarity is interpretable. |
| Chunking | **langchain-text-splitters** | `RecursiveCharacterTextSplitter` — splits on paragraph and heading boundaries before falling back to characters. |
| Schema & validation | **Pydantic 2.13** | Defines the judge's verdict shape and validates whatever the model returns. |
| Local inference | **LM Studio** | Serves an OpenAI-compatible API on `localhost:1234`, so `langchain-openai` talks to a local model with no code change. Embeddings: `nomic-embed-text-v1.5`. Chat: `qwen3.5-9b`. |
| Cloud inference | **langchain-anthropic / langchain-openai** | Swapped by environment variable, not by code. |
| Tests | **pytest 9** | Runs with no API keys and no local server. |

### LangChain concepts exercised

- **Provider abstraction** — one factory, three backends (`config.py`)
- **`@tool`** — a Python function becomes a model-callable tool; its docstring is the prompt (`tools.py`)
- **`create_agent`** — the agent loop, with `ToolCallLimitMiddleware` and a custom `wrap_tool_call` hook (`agent.py`)
- **`with_structured_output`** — typed model output, plus the manual fallback for backends that don't support it (`juez.py`)
- **Task-prefixed embeddings** — `search_document:` / `search_query:`, required by nomic-class models and silently degrading without them (`config.py`)

---

## Three engineering decisions worth explaining

**Cosine distance is forced explicitly.** Chroma defaults to L2, whose values
have no ceiling and can't be read as "how similar is this". Without
`hnsw:space=cosine` the whole measurement is meaningless.

**Failures are never reported as findings.** When the judge returns nothing
usable, the question is marked `ERROR` and **excluded from the score** — not
counted as a gap. Counting our own failures as the customer's documentation
problems would inflate the result with noise that looks like signal.

**The score is weighted by frequency, not question count.** A gap asked 142
times outweighs one asked 16 times. That weighting is what turns a vanity metric
into a prioritised work queue.

---

## Working around a leaky abstraction

LangChain's `with_structured_output` is the right way to get typed output from a
model. All three of its methods failed against LM Studio, each differently:

| Method | Failure |
|---|---|
| `json_mode` | LM Studio only accepts `json_schema` or `text` |
| `json_schema` | Response arrives without the `parsed` field LangChain expects |
| `function_calling` | Forcing a specific tool via object `tool_choice` is rejected |

"OpenAI-compatible" is compatible up to a point. The fallback is to request JSON
in the prompt and parse it ourselves — tolerant of `<think>` blocks, code
fences, and surrounding prose. Covered by tests. `JUEZ_MODO=structured` restores
the elegant path for cloud providers that support it.

Reasoning models add a second trap: they spend output tokens thinking before
answering. Budget too tightly and they return an empty string, having reasoned
until they ran out. Setting `JUEZ_MAX_TOKENS` too low silently broke every
verdict in a full run.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH=src
```

Put `.md` / `.txt` files in `docs/` and a CSV in `data/` with a `pregunta`
column (and optionally `frecuencia`), then:

```bash
python -m rag_agent.ingest
python -m rag_agent.reporte data/preguntas.csv
python -m rag_agent.cli              # chat over the same corpus
```

**Fully local:** load a chat model and an embedding model in LM Studio, start the
server, keep `LLM_PROVIDER=local`.

**Cloud:** set `LLM_PROVIDER=anthropic` and `EMBEDDINGS_PROVIDER=openai` —
Anthropic has no embeddings API, so the two are independent axes by design.

### Tests

Run with no API keys and no local server — fake deterministic embeddings and a
scripted chat model verify the wiring end to end.

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

---

## Known limitations

- **Sequential judging.** One call per question, no batching or caching. A
  200-question run against a local 9B takes a long time.
- **Judge boundary is a judgement call.** Whether "the rules of changing plans"
  answers "how do I change plans" is a product decision encoded in a prompt, not
  an objective fact. The prompt is in `juez.py` and is meant to be tuned.
- **Ingest handles `.md` / `.txt` only.** PDF means adding a loader.
- **Pure vector search, no reranking or hybrid BM25.** The ceiling shows on
  larger corpora.
- **Questions must be supplied.** The tool measures against real demand; it
  can't invent what your users would have asked.

## License

MIT
