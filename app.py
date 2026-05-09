import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import load_pdf, chunk_documents, build_vectorstore, load_vectorstore, build_qa_chain
from vision_pipeline import describe_image, ask_with_image

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Page config
st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title("Multimodal RAG Chatbot")
st.caption("Upload a PDF and optionally an image, then ask questions about both.")

# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "image_path" not in st.session_state:
    st.session_state.image_path = None

# Sidebar 
with st.sidebar:
    st.header("Upload Files")

    # PDF upload
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf_file:
        if st.button("Process PDF", use_container_width=True):
            with st.spinner("Reading and indexing PDF..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name

                # Build pipeline
                docs    = load_pdf(tmp_path)
                chunks  = chunk_documents(docs)
                vs      = build_vectorstore(chunks)
                st.session_state.chain, st.session_state.retriever = build_qa_chain(vs)
                st.session_state.chat_history = []  # reset chat on new PDF
            st.success(f"Indexed {len(docs)} page(s), {len(chunks)} chunks")

    st.divider()

    # Image upload
    img_file = st.file_uploader("Upload an Image (optional)", type=["jpg", "jpeg", "png", "webp"])
    if img_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img_file.name.split('.')[-1]}") as tmp:
            tmp.write(img_file.read())
            st.session_state.image_path = tmp.name
        st.image(img_file, caption="Uploaded Image", use_container_width=True)
        if st.button("Remove Image", use_container_width=True):
            st.session_state.image_path = None
            st.rerun()

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main chat area 
if not st.session_state.chain:
    st.info("Upload and process a PDF to get started.")
else:
    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**Page {s['page']}:** {s['snippet']}...")
            if msg["role"] == "assistant" and "image_desc" in msg:
                with st.expander("Image Description"):
                    st.markdown(msg["image_desc"])

    # Chat input
    question = st.chat_input("Ask a question about your document...")
    if question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.image_path:
                    result = ask_with_image(
                        st.session_state.chain,
                        st.session_state.retriever,
                        question,
                        st.session_state.image_path
                    )
                    image_desc = result["image_description"]
                else:
                    from rag_pipeline import ask
                    result = ask(st.session_state.chain, st.session_state.retriever, question)
                    image_desc = None

            st.markdown(result["answer"])

            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"**Page {s['page']}:** {s['snippet']}...")

            if image_desc:
                with st.expander("Image Description"):
                    st.markdown(image_desc)

        # Save to history
        history_entry = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        }
        if image_desc:
            history_entry["image_desc"] = image_desc
        st.session_state.chat_history.append(history_entry)