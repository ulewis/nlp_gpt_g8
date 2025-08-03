# -----------------------------------------------------------------------------
# streamlit_app.py — Buscador inteligente de concursos públicos (RAG + GPT‑4)
# -----------------------------------------------------------------------------
# Este script ofrece una interfaz Streamlit para consultar un vectorstore Chroma
# de contratos públicos. Incluye:
#   • Reformulación e identificación de intención (código vs servicio)
#   • Búsqueda por regex o búsqueda semántica (embeddings OpenAI)
#   • Validación / rerank de resultados con GPT‑4
#   • Tabla interactiva + visualización 3D opcional (PCA)
#   • Todo configurable desde la sidebar (vectorstore y OPENAI_API_KEY)
# -----------------------------------------------------------------------------

from __future__ import annotations
import importlib, sys, os, re, json
from typing import List, Dict, Any, Tuple

# --- Parche sqlite3 para entornos sin la librería nativa ---------------------
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = importlib.import_module("pysqlite3")
    sys.modules["sqlite3.dbapi2"] = sys.modules["sqlite3"]
except ImportError:
    pass

# --------------------------- Dependencias externas ---------------------------

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --------------------------- Constantes globales -----------------------------

EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1024           # Reducimos tamaño para ahorrar memoria
MAX_TOP_K         = 20             # Límite superior de resultados a mostrar
DEFAULT_TOP_K     = 5

# =========================== INTERFAZ DE USUARIO =============================

st.set_page_config(page_title="Buscador Concursos Públicos", layout="wide")

# --- Sidebar ----------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
vectorstore_path = st.sidebar.text_input("Ruta del vectorstore", "./chroma")
api_key          = st.sidebar.text_input("OPENAI_API_KEY", type="password")
show_pca         = st.sidebar.checkbox("Mostrar visualización 3D (PCA)")

st.sidebar.markdown(
    """🧠 **Modelo embeddings**: `text-embedding-3-small`  
📏 **Dimensión (slice)**: `1024`"""
)

# --- Main -------------------------------------------------------------------

st.title("🔎 Buscador inteligente de contratos públicos")
query = st.text_input("¿Qué tipo de contrato público deseas buscar?")
col_btn, col_k = st.columns([1, 1])
with col_k:
    top_k = st.number_input("Top‑K", 1, MAX_TOP_K, value=DEFAULT_TOP_K, step=1)
with col_btn:
    do_search = st.button("Buscar", use_container_width=True)

# ===================== EMBEDDINGS / VECTORSTORE HELPERS ======================

def sliced_embedder_factory(api_key: str):
    """Devuelve un wrapper de OpenAIEmbeddings recortado a EMBEDDING_DIM."""
    os.environ["OPENAI_API_KEY"] = api_key  # Garantiza que langchain lo vea
    base = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    class _Wrapper:
        def embed_documents(self, texts):
            return [v[:EMBEDDING_DIM] for v in base.embed_documents(texts)]
        def embed_query(self, text):
            return base.embed_query(text)[:EMBEDDING_DIM]
    return _Wrapper()

# =========================== PROMPTS & CHAINS ================================

prompt_reformulador = PromptTemplate.from_template(
    """
Eres un asistente especializado exclusivamente en encontrar contratos públicos de bienes o servicios.

1. Reformula la consulta del usuario para que sea clara y útil para una búsqueda semántica.
2. Detecta la intención:
   • "busqueda por codigo" si contiene un código de contratación.
   • "busqueda por servicio" si describe un bien o servicio.
   • "fuera del dominio" si no es contratación pública.
   • "no entendida" si no puedes determinar.

Devuelve **solo** JSON:
{{
  "consulta_reformulada": "...",
  "intencion": "..."
}}

Entrada del usuario:
"{entrada}"
"""
)
parser_reform = JsonOutputParser()

prompt_validador = PromptTemplate.from_template(
    """
Eres un evaluador que analiza si un concurso público está relacionado con la consulta:
"{consulta}"

Para cada concurso listado responde con:
- "relevancia": Directamente / Indirecto / No Relacionado.
- "justificacion": breve razón.

Devuelve JSON:
[
  {{"codigo": "...", "relevancia": "...", "justificacion": "..."}},
  ...
]

{concursos}
"""
)
parser_validador = JsonOutputParser()

# ----- Builder de Chains (se crea después de tener api_key) -----------------

def build_chains(api_key: str):
    llm_reform  = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
    llm_valida  = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
    return (
        prompt_reformulador | llm_reform | parser_reform,
        prompt_validador    | llm_valida | parser_validador,
    )

# ============================= SEARCH HELPERS ===============================

# --- Regex extracción de códigos -------------------------------------------

def extraer_codigo_desde_texto(texto: str) -> str:
    patrones = [
        r"\b[A-Z]{0,4}-?\d{2,}-\d{4}(?:-[A-Z0-9]+)*\b",  # CM-60-2025-OEDI
        r"\b\d{2,}-\d{4}\b",                             # 60-2025
        r"\b\d{4}-[A-Z]{2,}\b",                          # 2025-UNPRG
        r"\b[A-Z]{2,}-\d{4}\b",                          # UNPRG-2025
    ]
    for pat in patrones:
        if (m := re.search(pat, texto.upper())):
            return m.group(0)
    return texto.strip()

# --- Búsqueda por código ----------------------------------------------------

def buscar_por_codigo(codigo: str, vs: Chroma) -> List[Dict[str, Any]]:
    patron = re.compile(re.escape(codigo.strip()), re.IGNORECASE)
    data   = vs._collection.get(include=["documents", "metadatas"])
    resultados = []
    for meta, doc in zip(data["metadatas"], data["documents"]):
        cod = meta.get("codigo", "").strip()
        if patron.search(cod):
            resultados.append({
                "codigo":      cod,
                "entidad":     meta.get("entidad", "N/A"),
                "estado":      meta.get("estado", "N/A"),
                "publicacion": meta.get("publicacion", "N/A"),
                "link":        meta.get("link", "N/A"),
                "similitud":   1.0,
                "distancia":   0.0,
                "descripcion": doc[:500] + ("..." if len(doc) > 500 else ""),
            })
    return resultados

# --- Búsqueda semántica -----------------------------------------------------

def generar_vector_consulta(texto: str, embedder) -> np.ndarray:
    return np.array(embedder.embed_query(texto)).reshape(1, -1)


def buscar_top_k(vs: Chroma, q_vec: np.ndarray, k: int = 5):
    data = vs._collection.get(include=["embeddings", "documents", "metadatas"])
    emb_valid, docs_valid, metas_valid = [], [], []
    for emb, doc, meta in zip(data["embeddings"], data["documents"], data["metadatas"]):
        try:
            vec = np.array(emb, dtype=np.float32)
            if vec.ndim == 1 and vec.size == EMBEDDING_DIM and not np.isnan(vec).any():
                emb_valid.append(vec)
                docs_valid.append(doc)
                metas_valid.append(meta)
        except Exception:
            continue
    if not emb_valid:
        return []
    sims = cosine_similarity(q_vec, np.vstack(emb_valid))[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    return [(metas_valid[i], docs_valid[i], float(sims[i])) for i in top_idx]


def enriquecer_resultados(top: List[Tuple[Dict[str, Any], str, float]]):
    enr = []
    for rank, (meta, doc, score) in enumerate(top, start=1):
        enr.append({
            "ranking":     rank,
            "codigo":      meta.get("codigo", "N/A"),
            "entidad":     meta.get("entidad", "N/A"),
            "estado":      meta.get("estado", "N/A"),
            "publicacion": meta.get("publicacion", "N/A"),
            "link":        meta.get("link", "N/A"),
            "similitud":   round(score, 4),
            "distancia":   round(1 - score, 4),
            "descripcion": doc[:500] + ("..." if len(doc) > 500 else ""),
        })
    return enr

# ========================== EJECUCIÓN DE BÚSQUEDA ============================

if do_search:
    # --- Validaciones previas ---------------------------------------------
    if not api_key.strip():
        st.warning("Por favor ingresa tu clave de API.")
        st.stop()
    if not query.strip():
        st.warning("Por favor ingresa una consulta.")
        st.stop()

    # --- Cargar recursos ---------------------------------------------------
    embedder = sliced_embedder_factory(api_key)
    vs       = Chroma(persist_directory=vectorstore_path, embedding_function=embedder)
    try:
        st.sidebar.markdown(f"🗂️ **Documentos en colección:** {vs._collection.count()}")
    except Exception:
        st.sidebar.markdown("🗂️ **Documentos en colección:** ?")

    chain_reform, validador_chain = build_chains(api_key)

    # --- Paso 1: Reformulación + intención --------------------------------
    with st.spinner("Reformulando consulta …"):
        try:
            reformado = chain_reform.invoke({"entrada": query})
        except Exception as e:
            st.error(f"❌ Error al reformular: {e}")
            st.stop()

    consulta_reform = reformado.get("consulta_reformulada", "").strip()
    intencion       = reformado.get("intencion", "no entendida").lower()

    st.markdown(
        f"""**Consulta reformulada:** `{consulta_reform or '—'}`  \
**Intención detectada:** `{intencion}`"""
    )

    # --- Paso 2: Recuperación ---------------------------------------------
    resultados: List[Dict[str, Any]] = []
    if intencion == "fuera del dominio":
        st.info("La consulta no parece relacionada con contratación pública.")
        st.stop()

    elif intencion == "busqueda por codigo":
        codigo = extraer_codigo_desde_texto(consulta_reform or query)
        st.markdown(f"🔍 **Búsqueda por código:** `{codigo}`")
        resultados = buscar_por_codigo(codigo, vs)
        if not resultados:
            st.info("No se encontraron concursos con ese código.")
            st.stop()

    else:  # Búsqueda semántica (servicio u ambigua)
        st.markdown("🔍 **Búsqueda semántica**")
        q_vec = generar_vector_consulta(consulta_reform or query, embedder)
        top   = buscar_top_k(vs, q_vec, k=top_k)
        if not top:
            st.info("No se encontraron documentos relevantes.")
            st.stop()
        resultados = enriquecer_resultados(top)

    # --- Paso 3: Validación / Rerank con GPT‑4 -----------------------------
    with st.spinner("Validando relevancia con GPT‑4 …"):
        concursos_txt = "\n".join([f"- {c['codigo']}: {c['descripcion'][:120]}" for c in resultados])
        try:
            validados = validador_chain.invoke({
                "consulta":  consulta_reform or query,
                "concursos": concursos_txt,
            })
        except Exception as e:
            st.warning(f"No se pudo validar la relevancia: {e}")
            validados = []

    # Mezclamos resultados con relevancia
    actualizados: List[Dict[str, Any]] = []
    for c in resultados:
        cod_base = c["codigo"].strip().lower()
        match = next((v for v in validados if v["codigo"].strip().lower() in cod_base), None)
        if match and match["relevancia"].lower() != "no relacionado":
            c["relevancia"]    = match["relevancia"]
            c["justificacion"] = match["justificacion"]
            actualizados.append(c)

    if not actualizados:
        st.info("GPT‑4 consideró todos los resultados como no relacionados.")
        st.stop()

    # ------------------------- Mostrar resultados --------------------------
    df = pd.DataFrame(actualizados)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --------------------- Visualización PCA opcional ----------------------
    if show_pca and len(actualizados) >= 3 and intencion != "busqueda por codigo":
        ids = [c["codigo"] for c in actualizados]
        try:
            emb_data = vs._collection.get(ids=ids, include=["embeddings"])
            emb_valid = [e for e in emb_data["embeddings"] if isinstance(e, list) and len(e) == EMBEDDING_DIM]
        except Exception:
            emb_valid = []

        if len(emb_valid) >= 3:
            pts3d = PCA(n_components=3).fit_transform(np.array(emb_valid))
            fig = go.Figure(go.Scatter3d(
                x=pts3d[:, 0], y=pts3d[:, 1], z=pts3d[:, 2],
                mode="markers+text",
                text=[f"{c['codigo']}<br>{c['similitud']:.3f}" for c in actualizados[:len(pts3d)]],
                marker=dict(size=5)
            ))
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se pudo generar la gráfica 3D (menos de 3 embeddings válidos).")

# ============================= PRUEBAS BÁSICAS ===============================

if __name__ == "__main__":
    # Pruebas rápidas de utilidades (no Streamlit)
    assert extraer_codigo_desde_texto("Necesito CM-60-2025-OEDI") == "CM-60-2025-OEDI"
    assert extraer_codigo_desde_texto("Busco 60-2025") == "60-2025"
    assert extraer_codigo_desde_texto("2025-UNPRG") == "2025-UNPRG"
    assert extraer_codigo_desde_texto("UNPRG-2025") == "UNPRG-2025"
    print("✔️ Tests locales de extracción de códigos superados.")

