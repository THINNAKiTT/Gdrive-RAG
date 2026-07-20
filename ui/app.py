import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.main import sync_knowledge_base
from src.rag.orchestrator import RAGOrchestrator

st.set_page_config(
    page_title="GDrive-RAG-Chatbot",
    layout="centered",
)
st.title("GDrive-RAG-Chat")
st.caption("Synchronized with Google Drive")

@st.cache_resource
def setup_rag_sys():
    orchestrator = RAGOrchestrator()
    return orchestrator.get_query_engine()

try:
    with st.spinner("Initializing database and loading RAG..."):
        query_engine = setup_rag_sys()
except Exception as e:
    st.error(f"Initialization Failed: {str(e)}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("References"):
                for citation in message["citations"]:
                    st.caption(citation)

if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Syncing with the database."):
            sync_knowledge_base()   
        with st.spinner("Searching documents and generating response..."):
            response = query_engine.query(user_query)
            answer_text = getattr(response, "response", str(response))

            citations = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    meta = node.node.metadata
                    file_name = meta.get("file_name", "Unknown File")
                    page_num = meta.get("page_number", "N/A")
                    url = meta.get("web_view_link", "#")
                    score = node.score if node.score else 0.0

                    citation_text = f"[{file_name} (Page {page_num})]({url}) - Score : {score:.2f}"
                    if citation_text not in citations:
                        citations.append(citation_text)

            st.markdown(answer_text)
            if citations:
                with st.expander("References"):
                    for citation in citations:
                        st.caption(citation)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "citations": citations
            })