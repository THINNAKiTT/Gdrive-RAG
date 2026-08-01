import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.storage.sync_lock import SyncLock
from src.rag.orchestrator import RAGOrchestrator
from src.rag.query_rewriter import QueryRewriter
from src.storage.chat_history import ChatHistoryStore
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

@st.cache_resource
def setup_query_rewriter():
    return QueryRewriter(max_turns=6)


@st.cache_resource
def setup_chat_store():
    return ChatHistoryStore()

sync_lock = SyncLock()
chat_store = setup_chat_store()

try:
    with st.spinner("Initializing database and loading RAG..."):
        query_engine = setup_rag_sys()
        query_rewriter = setup_query_rewriter()
except Exception as e:
    st.error(f"Initialization Failed: {str(e)}")
    st.stop()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

def _create_new_chat():
    st.session_state.current_session_id = chat_store.generate_pending_session_id()


def _switch_session(session_id):
    st.session_state.current_session_id = session_id


def _delete_session(session_id, was_active):
    chat_store.delete_session(session_id)
    if was_active:
        st.session_state.current_session_id = None

with st.sidebar:
    st.header("Chats")

    st.button(
        "+ New Chat",
        key="new_chat_button",
        use_container_width=True,
        on_click=_create_new_chat,
    )

    st.divider()

    sessions = chat_store.list_sessions()

    if not sessions:
        st.caption("No chats yet. Start a new one above.")
    else:
        for session in sessions:
            is_active = session["id"] == st.session_state.current_session_id
            col1, col2 = st.columns([5, 1])
            with col1:
                label = session["title"] or "Untitled chat"
                st.button(
                    label,
                    key=f"session_{session['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    on_click=_switch_session,
                    args=(session["id"],),
                )
            with col2:
                st.button(
                    "",
                    icon=":material/delete_outline:",
                    key=f"delete_{session['id']}",
                    on_click=_delete_session,
                    args=(session["id"], is_active),
                )

if st.session_state.current_session_id is None: 
    existing_sessions = chat_store.list_sessions()
    if existing_sessions:
        st.session_state.current_session_id = existing_sessions[0]["id"]
    else:
        st.session_state.current_session_id = chat_store.generate_pending_session_id()

current_session_id = st.session_state.current_session_id

messages = chat_store.get_messages(current_session_id)
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("References"):
                for citation in message["citations"]:
                    st.markdown(citation)

if user_query := st.chat_input("Ask a question about your documents..."):
    chat_store.add_message(current_session_id, "user", user_query)
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

        recent_turns = chat_store.get_recent_turns(current_session_id, max_turns=6)
        recent_turns = recent_turns[:-1] if recent_turns else recent_turns
        rewritten_query = query_rewriter.rewrite(user_query, recent_turns)
        
        with st.spinner("Searching documents and generating response..."):
            try:
                query_with_resilience = with_resilience(query_engine.query)
                response = query_with_resilience(rewritten_query)
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
            if response and hasattr(response, "source_nodes"):
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

            chat_store.add_message(current_session_id, "assistant", answer_text, citations=citations)