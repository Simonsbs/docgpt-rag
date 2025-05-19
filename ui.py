import streamlit as st
from app.utils.config_loader import load_config
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient
from app.vectorstore.chroma_store import VectorStore
from app.utils.history_logger import log_query
import difflib
import json
import os
import time

# --- Init
os.makedirs("docs", exist_ok=True)
os.makedirs("sessions", exist_ok=True)
st.set_page_config(page_title="Chat with Simon", layout="centered")
st.title("🤖 Chat with Simon")

# --- Session state init
if "chat" not in st.session_state:
    st.session_state.chat = []
if "log" not in st.session_state:
    st.session_state.log = []
if "chunk_score_threshold" not in st.session_state:
    st.session_state.chunk_score_threshold = 0
if "model_provider" not in st.session_state:
    st.session_state.model_provider = "Ollama"

# --- Load vector store
vs = VectorStore()

# --- Sidebar
with st.sidebar:
    st.markdown("### 👤 About Simon")
    st.write("You're chatting with an AI trained on Simon Stirling’s background and work.")
    st.markdown("Ask me about my experience, projects, philosophy, career path, skills, or goals.")

    st.markdown("---")
    selected_model = st.selectbox(
        "Model Provider:",
        ["Ollama", "OpenAI"],
        index=["Ollama", "OpenAI"].index(st.session_state.model_provider)
    )
    st.session_state.model_provider = selected_model
    st.slider("Minimum chunk match % to display:", 0, 100, st.session_state.chunk_score_threshold, key="chunk_score_threshold")

    st.markdown("---")
    st.markdown("### ℹ️ System Log")
    for msg in st.session_state.log[-10:]:
        st.text(msg)

    st.markdown("---")
    if st.button("📏 Save Session"):
        with open("sessions/session.json", "w") as f:
            json.dump(st.session_state.chat, f)
        st.success("Session saved.")
    if st.button("📂 Load Session"):
        try:
            with open("sessions/session.json", "r") as f:
                st.session_state.chat = json.load(f)
            st.success("Session loaded.")
        except:
            st.error("No saved session found.")

# --- User input
query = st.chat_input("Ask me anything about myself...")
history = st.session_state.chat[:-2] if query else st.session_state.chat

# --- On submit
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat.append(("user", query))
    st.session_state.log.append(f"User asked: {query}")

    config = load_config()
    llm = OllamaClient() if st.session_state.model_provider == "Ollama" else OpenAIClient()

    # --- Retrieval step
    t0 = time.time()
    results = vs.query(query)
    retrieval_time = time.time() - t0

    sources = results
    context = "\n".join(doc for _, doc in results)
    prompt = (
        f"You are Simon Stirling, answering questions about yourself based only on the provided context.\n"
        f"Context:\n{context}\n\n"
        f"User question: {query}\n"
        f"Answer in first-person as Simon."
    )
    st.session_state.log.append(f"Retrieved {len(results)} chunks")

    t1 = time.time()
    # --- LLM stream
    with st.chat_message("assistant"):
        streamed_output = llm.stream(prompt)
        response_container = st.empty()
        full_response = ""

        for token in streamed_output:
            full_response += token
            response_container.markdown(full_response.strip())

        st.session_state.chat.append(("assistant", full_response))
        log_query(query, full_response)
        st.session_state.log.append("Answer streamed")

        llm_time = time.time() - t1
        st.session_state.log.append(f"RAG time: {retrieval_time:.2f}s | LLM time: {llm_time:.2f}s")

        with st.expander("📊 RAG Pipeline Visualizer"):
            st.markdown(f"- **Embedding + Vector Search Time:** `{retrieval_time:.2f}s`")
            st.markdown(f"- **LLM Response Time:** `{llm_time:.2f}s`")
            st.markdown(f"- **Retrieved Chunks:** `{len(results)}`")
            st.markdown("### 🧠 Retrieved Chunk IDs:")
            for doc_id, _ in sources:
                st.code(doc_id, language="bash")
            st.markdown("### 📝 Final Prompt to LLM:")
            st.code(prompt.strip(), language="markdown")

            st.markdown("### 📈 RAG Flow Diagram")
            st.graphviz_chart(f"""
                digraph RAG {{
                    Q [label="User Query"];
                    E [label="Embed + Vector Search\n{retrieval_time:.2f}s"];
                    R [label="Top K Chunks"];
                    P [label="Prompt Construction"];
                    L [label="LLM Response\n{llm_time:.2f}s"];
                    A [label="Final Answer"];

                    Q -> E -> R -> P -> L -> A;
                }}
            """)

        with st.expander("📋 Source Chunks"):
            threshold = st.session_state.chunk_score_threshold
            for doc_id, chunk in sources:
                score = int(difflib.SequenceMatcher(None, chunk, full_response).ratio() * 100)
                if score >= threshold:
                    st.markdown(f"**{doc_id}** (match: {score}%)")
                    st.code(chunk.strip(), language="markdown")

        with st.expander("📅 Export chat"):
            full_chat = "\n\n".join(f"**{role.capitalize()}**: {msg}" for role, msg in st.session_state.chat)
            st.download_button("Download .md", full_chat, file_name="chat_with_simon.md")

# --- Replay chat history (excluding latest)
for role, msg in history:
    with st.chat_message(role):
        st.markdown(msg)
