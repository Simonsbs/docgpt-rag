import streamlit as st
from app.utils.config_loader import load_config
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient
from app.vectorstore.chroma_store import VectorStore
from app.utils.history_logger import log_query
import json
import os
import time

# --- Define a chat message object
class ChatEntry:
    def __init__(self, sender: str, content: str, metadata: dict = None):
        self.sender = sender  # 'user' or 'assistant'
        self.content = content
        self.metadata = metadata or {}

# --- Init directories
os.makedirs("docs", exist_ok=True)
os.makedirs("sessions", exist_ok=True)
st.set_page_config(page_title="Chat with Simon", layout="wide")
st.title("🤖 Chat with Simon")

# --- Session state init
if "chat_entries" not in st.session_state:
    st.session_state.chat_entries = []  # list[ChatEntry]
if "selected_meta" not in st.session_state:
    st.session_state.selected_meta = None
if "log" not in st.session_state:
    st.session_state.log = []
if "chunk_score_threshold" not in st.session_state:
    st.session_state.chunk_score_threshold = 0
if "model_provider" not in st.session_state:
    st.session_state.model_provider = "Ollama"

# --- Load vector store
vs = VectorStore()

# --- Layout: chat and metadata panels
col_chat, col_meta = st.columns([3, 1])

# --- Metadata panel
with col_meta:
    st.markdown("### 📋 Response Metadata")
    meta = st.session_state.selected_meta
    if meta:
        st.markdown(f"**Retrieval time:** {meta['retrieval_time']:.2f}s")
        st.markdown(f"**Model time:** {meta['llm_time']:.2f}s")
        st.markdown("**Retrieved Chunks:**")
        for cid in meta['chunk_ids']:
            st.text(f"- {cid}")
        st.markdown("**Prompt:**")
        st.code(meta['prompt'], language="markdown")
    else:
        st.info("Click 'Show metadata' on a response to view details.")

# --- Chat UI
with col_chat:
    # Sidebar controls
    st.sidebar.markdown("### 👤 About Simon")
    st.sidebar.write("AI trained on Simon Stirling’s background.")
    st.sidebar.markdown("---")
    provider = st.sidebar.selectbox(
        "Model Provider:", ["Ollama", "OpenAI"],
        index=["Ollama", "OpenAI"].index(st.session_state.model_provider)
    )
    st.session_state.model_provider = provider
    st.sidebar.slider(
        "Min chunk match %:", 0, 100,
        st.session_state.chunk_score_threshold,
        key="chunk_score_threshold"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ System Log")
    for msg in st.session_state.log[-10:]:
        st.sidebar.text(msg)

    # Display existing chat
    for idx, entry in enumerate(st.session_state.chat_entries):
        with st.chat_message(entry.sender):
            st.markdown(entry.content)
        if entry.sender == "assistant":
            if st.button("Show metadata", key=f"hist_meta_{idx}"):
                st.session_state.selected_meta = entry.metadata

    # User input
    query = st.chat_input("Ask me anything about myself...")

    # On submit
    if query:
        # User entry
        user_entry = ChatEntry(sender="user", content=query)
        st.session_state.chat_entries.append(user_entry)
        st.session_state.log.append(f"User: {query}")
        with st.chat_message("user"):
            st.markdown(query)

        # Select LLM
        llm = OllamaClient() if provider == "Ollama" else OpenAIClient()

        # Retrieval
        t0 = time.time()
        results = vs.query(query)
        retrieval_time = time.time() - t0
        chunk_ids = [cid for cid, _ in results]
        context = "\n".join(doc for _, doc in results)
        prompt = (
            f"You are Simon Stirling, answer based on provided context.\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        st.session_state.log.append(f"Retrieved {len(results)} chunks in {retrieval_time:.2f}s")

        # Assistant streaming
        full_response = ""
        t1 = time.time()
        with st.chat_message("assistant"):
            # stream into the bubble
            response_container = st.empty()
            for token in llm.stream(prompt):
                full_response += token
                response_container.markdown(full_response)

        llm_time = time.time() - t1
        st.session_state.log.append(f"Assistant answered in {llm_time:.2f}s")

        # Assistant entry
        assistant_entry = ChatEntry(
            sender="assistant",
            content=full_response,
            metadata={
                'retrieval_time': retrieval_time,
                'llm_time': llm_time,
                'chunk_ids': chunk_ids,
                'prompt': prompt
            }
        )
        st.session_state.chat_entries.append(assistant_entry)
        st.session_state.selected_meta = assistant_entry.metadata
        log_query(query, full_response)
        
        # Metadata button below bubble
        if st.button("Show metadata", key=f"meta_{len(st.session_state.chat_entries)}"):
            st.session_state.selected_meta = assistant_entry.metadata
