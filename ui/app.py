import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.rag.orchestrator import RAGOrchestrator
from src.storage.sync_lock import SyncLock
from src.utils.resilience import with_resilience, CircuitOpenError

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

sync_lock = SyncLock()

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
        if sync_lock.is_locked():
            with st.spinner("Waiting for an in-progress sync to finish..."):
                caught_up = sync_lock.wait_until_free(timeout_seconds=30.0)
            if not caught_up:
                st.warning(
                    "Sync is taking longer than expected -- answering with "
                    "the current index, which may not include the very "
                    "latest Drive changes."
                )
        
        with st.spinner("Searching documents and generating response..."):
            try:
                query_with_resilience = with_resilience(query_engine.query)
                response = query_engine.query(user_query)
                answer_text = getattr(response, "response", str(response))
            except CircuitOpenError as e:
                st.error(
                    "AI server seems to be down right now, so I can't answer "
                    "this question. Please check that AI server is running "
                    f"and try again shortly. ({e})"
                )
                st.stop()
            except Exception as e:
                st.error(f"Something went wrong while generating a response: {e}")
                st.stop()

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
                        st.markdown(citation)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "citations": citations
            })