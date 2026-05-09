import os
import base64
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

# Encode image to base64
def encode_image(image_path: str) -> tuple[str, str]:
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    mime = mime_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime

# Describe image using Gemini Vision
def describe_image(image_path: str, context_hint: str = "") -> str:
    data, mime = encode_image(image_path)

    prompt = f"""Describe this image in detail. Be specific about:
- What is shown (objects, people, text, diagrams, charts, etc.)
- Any text visible in the image
- Spatial relationships and layout
- Colors, quantities, and any measurable details
{f'Context hint: {context_hint}' if context_hint else ''}
Your description will be used to answer questions, so be thorough."""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime),
            types.Part.from_text(text=prompt)
        ]
    )
    description = response.text
    print(f"[✓] Image described ({len(description)} chars)")
    return description

# Ask question using both image + RAG context 
def ask_with_image(qa_chain, retriever, question: str, image_path: str) -> dict:
    # Get image description
    image_desc = describe_image(image_path)

    # Augment question with image context
    augmented_question = f"""
The user has also provided an image. Here is a detailed description of it:

IMAGE DESCRIPTION:
{image_desc}

Now answer this question using both the document context AND the image description above:
{question}
"""
    # Retrieve relevant doc chunks
    sources = retriever.invoke(question)
    doc_context = "\n\n".join([d.page_content for d in sources])

    final_prompt = f"""Answer the question using the document context and image description below.
If the answer isn't in either, say "I don't know."

DOCUMENT CONTEXT:
{doc_context}

IMAGE DESCRIPTION:
{image_desc}

QUESTION: {question}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[types.Part.from_text(text=final_prompt)]
    )

    return {
        "answer": response.text,
        "image_description": image_desc,
        "sources": [
            {"page": d.metadata.get("page"), "snippet": d.page_content[:200]}
            for d in sources
        ]
    }

# Quick test 
if __name__ == "__main__":
    from rag_pipeline import load_pdf, chunk_documents, build_vectorstore, build_qa_chain

    PDF_PATH   = "example.pdf"
    IMAGE_PATH = "example.png"  # drop any image here to test

    # Build RAG pipeline
    docs    = load_pdf(PDF_PATH)
    chunks  = chunk_documents(docs)
    vs      = build_vectorstore(chunks)
    chain, retriever = build_qa_chain(vs)

    # Test image description alone
    print("\n── Image Description ──")
    desc = describe_image(IMAGE_PATH)
    print(desc)

    # Test combined question
    print("\n── Combined Q&A ──")
    result = ask_with_image(chain, retriever, "What does the image show and how does it relate to the document?", IMAGE_PATH)
    print(f"A: {result['answer']}")