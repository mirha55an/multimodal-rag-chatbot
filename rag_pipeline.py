import os
from dotenv import load_dotenv
import fitz  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# PDF to raw text
def load_pdf(pdf_path: str) -> list[Document]:
    doc = fitz.open(pdf_path)
    docs = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page": i + 1, "source": pdf_path}
            ))
    print(f"Loaded {len(docs)} pages from '{pdf_path}'")
    return docs

# Chunk the text 
def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# Embed + store in ChromaDB 
def build_vectorstore(chunks: list[Document], persist_dir: str = "chroma_db") -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Vector store built and saved to '{persist_dir}'")
    return vectorstore

# Load existing vectorstore (skip re-embedding) 
def load_vectorstore(persist_dir: str = "chroma_db") -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Build QA chain (LCEL)
def build_qa_chain(vectorstore: Chroma):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# Ask a question 
def ask(chain, retriever, question: str) -> dict:
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": [
            {"page": d.metadata.get("page"), "snippet": d.page_content[:200]}
            for d in sources
        ]
    }

# Quick test 
if __name__ == "__main__":
    PDF_PATH ="example.pdf"

    docs   = load_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    vs     = build_vectorstore(chunks)
    chain, retriever = build_qa_chain(vs)

    q = "What is this document about?"
    out = ask(chain, retriever, q)
    print(f"\nQ: {q}\nA: {out['answer']}\n")
    print("Sources:")
    for s in out["sources"]:
        print(f"  Page {s['page']}: {s['snippet']}...")