"""
Test end-to-end SIN API keys ni servidor local.

Truco: embeddings determinísticos falsos + un chat model falso que devuelve
una secuencia fija de mensajes. Así se verifica el CABLEADO (¿se llama la
tool? ¿vuelve el ToolMessage? ¿corre el middleware?) sin depender de red.

Esto es lo que separa un repo de demo de uno que alguien se toma en serio.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ToolCapableFake(FakeMessagesListChatModel):
    """FakeMessagesListChatModel no implementa bind_tools; se lo agregamos."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    import rag_agent.config as cfg
    import rag_agent.ingest as ing

    monkeypatch.setattr(cfg, "INDEX_DIR", tmp_path / "chroma")
    monkeypatch.setattr(ing, "INDEX_DIR", tmp_path / "chroma")
    monkeypatch.setattr(ing, "get_embeddings", lambda: DeterministicFakeEmbedding(size=256))

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "nota.md").write_text(
        "## Bleed\nEl bleed es el porcentaje de respuestas donde se cita a un "
        "competidor en lugar de la marca objetivo.",
        encoding="utf-8",
    )
    ing.ingest(docs)
    return tmp_path


def test_ingest_indexa_chunks(indexed):
    from rag_agent.tools import listar_fuentes

    assert "nota.md" in listar_fuentes.invoke({})


def test_retrieval_devuelve_fuente(indexed):
    from rag_agent.tools import buscar_en_documentos

    out = buscar_en_documentos.invoke({"query": "bleed"})
    assert "fuente: nota.md" in out


def test_agente_llama_la_tool_y_responde(indexed, monkeypatch):
    import rag_agent.agent as ag

    fake = ToolCapableFake(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "buscar_en_documentos", "args": {"query": "bleed"}, "id": "c1"}
                ],
            ),
            AIMessage(content="Es el % de respuestas con cita al competidor [nota.md]."),
        ]
    )
    monkeypatch.setattr(ag, "get_chat_model", lambda **kw: fake)

    agent = ag.build_agent(verbose=False)
    res = agent.invoke({"messages": [{"role": "user", "content": "¿Qué es el bleed?"}]})

    tipos = [type(m).__name__ for m in res["messages"]]
    assert tipos == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert "[nota.md]" in res["messages"][-1].content


# --- parser del juez: respuestas sucias que devuelven los modelos reales ----

@pytest.mark.parametrize(
    "crudo,esperado",
    [
        ('{"responde": true, "motivo": "esta"}', True),
        ('```json\n{"responde": false, "motivo": "no"}\n```', False),
        ('<think>pensando</think>\n{"responde": true, "motivo": "ok"}', True),
        ('Claro:\n```json\n{"responde": false, "motivo": "nada"}\n```\nEspero ayude.', False),
        ('<think>{"trampa": 1}</think> {"responde": true, "motivo": "x"}', True),
    ],
)
def test_extraer_json_tolera_basura(crudo, esperado):
    from rag_agent.juez import Dictamen, extraer_json

    assert Dictamen(**extraer_json(crudo)).responde is esperado


def test_extraer_json_falla_sin_json():
    from rag_agent.juez import extraer_json

    with pytest.raises(ValueError):
        extraer_json("no hay ningún objeto acá")
