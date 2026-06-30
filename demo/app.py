"""
Interface Streamlit pour ClinicalRAG — démo publique.
Déployée sur Hugging Face Spaces.
"""

import streamlit as st

from rag_demo import answer

# ─── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalRAG",
    page_icon="🩺",
    layout="centered",
)

# ─── En-tête ────────────────────────────────────────────────────────────────
st.title("🩺 ClinicalRAG")
st.caption(
    "Assistant de recherche clinique basé sur RAG — "
    "PubMed (811 articles) + Guidelines OMS (6 documents)"
)

with st.expander("ℹ️ Comment ça fonctionne ?"):
    st.markdown(
        """
        ClinicalRAG répond aux questions médicales en cherchant dans :
        - **811 articles PubMed** (8 thématiques : diabète, hypertension, asthme,
          dépression, cardiovasculaire, obésité, antibiorésistance, cancer)
        - **6 guidelines OMS** (recommandations cliniques officielles)

        Chaque réponse cite ses sources avec des références numérotées [n].
        Si aucune source pertinente n'est trouvée, le système le dit clairement
        plutôt que d'inventer — zéro hallucination par design.

        **Stack :** BAAI/bge-m3 (embeddings) · Groq llama-3.3-70b (LLM) ·
        ChromaDB (vector store) · Streamlit
        """
    )

st.divider()

# ─── Exemples de questions ───────────────────────────────────────────────────
st.markdown("**💡 Exemples de questions :**")
col1, col2 = st.columns(2)

EXAMPLES = [
    "What are the first-line drug classes for hypertension?",
    "What are the risk factors for type 2 diabetes?",
    "Quel est le mécanisme de résistance aux antibiotiques ?",
    "What lifestyle changes reduce cardiovascular disease risk?",
]

def set_question(q):
    st.session_state["question_input"] = q

with col1:
    st.button(EXAMPLES[0], on_click=set_question, args=(EXAMPLES[0],), use_container_width=True)
    st.button(EXAMPLES[2], on_click=set_question, args=(EXAMPLES[2],), use_container_width=True)
with col2:
    st.button(EXAMPLES[1], on_click=set_question, args=(EXAMPLES[1],), use_container_width=True)
    st.button(EXAMPLES[3], on_click=set_question, args=(EXAMPLES[3],), use_container_width=True)

st.divider()

# ─── Zone de saisie ─────────────────────────────────────────────────────────
question = st.text_input(
    "Posez votre question médicale :",
    key="question_input",
    placeholder="Ex : What are the WHO recommendations for hypertension treatment ?",
)

if st.button("🔍 Rechercher", type="primary", disabled=not question):
    with st.spinner("Recherche en cours (retrieval + génération)…"):
        result = answer(question)

    st.divider()

    # Réponse
    st.markdown("### 💬 Réponse")
    st.markdown(result["answer"])

    # Sources
    if result["sources"]:
        st.markdown(f"### 📚 Sources ({result['num_sources']})")
        for source in result["sources"]:
            st.markdown(f"- {source}")
    else:
        st.info("Aucune source pertinente trouvée dans la base documentaire.")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ ClinicalRAG est un outil de recherche documentaire, "
    "pas un substitut à un avis médical professionnel. "
    "Consultez toujours un professionnel de santé pour des décisions cliniques."
)
