# langgraph_bot.py ------------------------------------------------------------
import re, numpy as np
from typing import TypedDict, List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

# -------------------------- Estado ------------------------------------------
class AgentState(TypedDict):
    input: str
    consulta_reformulada: str
    intencion: str
    intentos_fuera_dominio: int
    concursos: List[Dict[str, Any]]

# ----------------- Auxiliar: extraer código ---------------------------------
def extraer_codigo_desde_texto(texto: str) -> str:
    texto = texto.upper()
    patrones = [
        r'\b[A-Z]{0,4}-?\d{2,}-\d{4}(?:-[A-Z0-9]+)*\b',
        r'\b\d{2,}-\d{4}\b',
        r'\b\d{4}-[A-Z]{2,}\b',
        r'\b[A-Z]{2,}-\d{4}\b',
    ]
    for patron in patrones:
        m = re.search(patron, texto)
        if m:
            return m.group(0)
    return texto.strip()

# --------------------------------------------------------------------------- #
#  ⬇️  Construye el grafo usando el vectorstore y el EMBEDDER que le pases    #
# --------------------------------------------------------------------------- #
def get_compiled_graph(vectorstore, embedder):
    """
    vectorstore: instancia de Chroma ya abierta.
    embedder:    objeto con .embed_query(text) que DEVUELVE el mismo largo
                 que los embeddings de tu store (p.ej. 1024).
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)  # OPENAI_API_KEY debe estar en el entorno

    # ---------------- Reformulador ----------------
    prompt_reform = PromptTemplate.from_template("""
Eres un asistente especializado en encontrar contratos públicos.

Tareas:
1) Reformula la consulta para búsqueda semántica.
2) Detecta intención: "busqueda por codigo", "busqueda por servicio",
   "no entendida" o "fuera del dominio".

Devuelve JSON:
{{"consulta_reformulada":"...","intencion":"..."}}

Entrada:
"{entrada}"
""")
    parser_reform = JsonOutputParser()
    chain_reform  = prompt_reform | llm | parser_reform

    def agente_reformulador(state: AgentState) -> AgentState:
        res = chain_reform.invoke({"entrada": state["input"]})
        return {**state,
                "consulta_reformulada": res.get("consulta_reformulada","").strip(),
                "intencion": res.get("intencion","").strip().lower()}

    # -------------- Búsqueda por código --------------
    def buscar_por_codigo(codigo: str):
        patron = re.compile(re.escape(codigo.strip()), re.IGNORECASE)
        data   = vectorstore._collection.get(include=["documents", "metadatas"])
        out    = []
        for meta, doc in zip(data["metadatas"], data["documents"]):
            cod = meta.get("codigo", "")
            if patron.search(cod):
                out.append({
                    "codigo": cod,
                    "entidad": meta.get("entidad","-"),
                    "estado": meta.get("estado","-"),
                    "publicacion": meta.get("publicacion","-"),
                    "link": meta.get("link","-"),
                    "similitud": 1.0,
                    "descripcion": doc[:500]+"…" if len(doc)>500 else doc
                })
        return out

    def nodo_busqueda_codigo(state: AgentState) -> AgentState:
        cod = extraer_codigo_desde_texto(state["consulta_reformulada"])
        return {**state, "concursos": buscar_por_codigo(cod)}

    # -------------- Búsqueda semántica --------------
    def buscar_semantica(texto: str, k: int = 5):
        # Usa el MISMO embedder que recorta al tamaño del store
        qvec = np.array(embedder.embed_query(texto)).reshape(1, -1)
        data = vectorstore._collection.get(include=["embeddings","documents","metadatas"])
        embs, docs, metas = [], [], []
        for e, d, m in zip(data["embeddings"], data["documents"], data["metadatas"]):
            if isinstance(e, list):
                e = np.array(e, dtype=np.float32)
                if e.ndim == 1 and e.size == qvec.shape[1] and not np.isnan(e).any():
                    embs.append(e); docs.append(d); metas.append(m)
        if not embs:
            return []
        mat  = np.vstack(embs)
        sims = cosine_similarity(qvec, mat)[0]
        idxs = sims.argsort()[-k:][::-1]
        res  = []
        for i in idxs:
            res.append({
                "codigo": metas[i].get("codigo","-"),
                "entidad": metas[i].get("entidad","-"),
                "estado": metas[i].get("estado","-"),
                "publicacion": metas[i].get("publicacion","-"),
                "link": metas[i].get("link","-"),
                "similitud": round(float(sims[i]),4),
                "descripcion": docs[i][:500]+"…" if len(docs[i])>500 else docs[i]
            })
        return res

    def flujo_busqueda_sem(state: AgentState) -> AgentState:
        return {**state, "concursos": buscar_semantica(state["consulta_reformulada"])}

    # -------------- “Validador” passthrough --------------
    def passthrough(state): return state

    # ----------------- Grafo ----------------------
    graph = StateGraph(AgentState)
    graph.add_node("Reform",        RunnableLambda(agente_reformulador))
    graph.add_node("BuscaCodigo",   RunnableLambda(nodo_busqueda_codigo))
    graph.add_node("BuscaSem",      RunnableLambda(flujo_busqueda_sem))
    graph.add_node("Validador",     RunnableLambda(passthrough))

    def branch(state):
        i = state["intencion"]
        if i == "busqueda por codigo":               return "BuscaCodigo"
        if i in ("fuera del dominio","no entendida"):return "__end__"
        return "BuscaSem"

    graph.add_conditional_edges("Reform", branch)
    graph.add_edge("BuscaCodigo", "Validador")
    graph.add_edge("BuscaSem",    "Validador")
    graph.set_entry_point("Reform")
    graph.set_finish_point("Validador")

    return graph.compile()
