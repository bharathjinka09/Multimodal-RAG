"""Multimodal RAG pipeline for the NovaCore FY2026 report.

This script mirrors the notebook logic and can be run directly as a Python script.
It loads keys from a local .env file if present, otherwise it prompts for them.
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from getpass import getpass

import fitz
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def load_environment_file() -> None:
    """Load environment variables from a local .env file if it exists."""
    load_dotenv(Path(__file__).with_name(".env"), override=False)


def ensure_api_key(name: str, label: str) -> str:
    value = os.getenv(name)
    if value:
        return value

    value = getpass(f"Enter {label}: ")
    os.environ[name] = value
    return value


load_environment_file()

GROQ_API_KEY = ensure_api_key("GROQ_API_KEY", "Groq API key")
PINECONE_API_KEY = ensure_api_key("PINECONE_API_KEY", "Pinecone API key")

PDF_PATH = Path("NovaCore_Multimodal_Company_Report_2026.pdf")
if not PDF_PATH.exists():
    fallback = Path("/mnt/data/NovaCore_Multimodal_Company_Report_2026.pdf")
    if fallback.exists():
        PDF_PATH = fallback

assert PDF_PATH.exists(), (
    "PDF not found. Keep NovaCore_Multimodal_Company_Report_2026.pdf "
    "in the same folder as this script."
)

TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PINECONE_INDEX_NAME = "novacore-multimodal-rag"
PINECONE_NAMESPACE = "fy2026-demo"

IMAGE_DIR = Path("novacore_extracted_images")
IMAGE_DIR.mkdir(exist_ok=True)

print("PDF:", PDF_PATH)
print("Pinecone index:", PINECONE_INDEX_NAME)
print("Pinecone namespace:", PINECONE_NAMESPACE)

# API clients

groq_client = Groq(api_key=GROQ_API_KEY)
text_llm = ChatGroq(model=TEXT_MODEL, temperature=0)


def image_to_data_uri(image_path: str | Path) -> str:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def summarize_visual(image_path: str | Path, page_number: int) -> str:
    image_data = image_to_data_uri(image_path)

    prompt = f"""
This visual was extracted from page {page_number} of the NovaCore FY2026 company report.

Describe the useful business information visible in the visual.

If it is a chart or graph:
- mention important values
- mention highest/lowest values
- mention the main trend

If it is a diagram:
- identify important components
- explain the flow or relationships

If it is a normal business image:
- describe the useful factual information

Keep the description concise and factual.
"""

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data},
                    },
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=500,
    )

    return response.choices[0].message.content.strip()


def extract_multimodal_documents(pdf_path: str | Path) -> list[Document]:
    pdf = fitz.open(str(pdf_path))
    documents: list[Document] = []
    extracted_xrefs: set[int] = set()

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        page_number = page_index + 1

        print(f"Processing page {page_number}...")

        text = page.get_text("text").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "modality": "text",
                        "source": Path(pdf_path).name,
                    },
                )
            )

        try:
            tables = page.find_tables().tables
            for table_number, table in enumerate(tables, start=1):
                df = table.to_pandas()
                if not df.empty:
                    table_text = df.to_markdown(index=False)
                    documents.append(
                        Document(
                            page_content=table_text,
                            metadata={
                                "page": page_number,
                                "modality": "table",
                                "table_number": table_number,
                                "source": Path(pdf_path).name,
                            },
                        )
                    )
        except Exception as error:
            print("Table extraction warning:", error)

        for image_number, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            if xref in extracted_xrefs:
                continue
            extracted_xrefs.add(xref)

            image_data = pdf.extract_image(xref)
            image_bytes = image_data["image"]
            image_extension = image_data["ext"]
            image_path = IMAGE_DIR / f"page_{page_number}_image_{image_number}.{image_extension}"
            image_path.write_bytes(image_bytes)

            try:
                summary = summarize_visual(image_path=image_path, page_number=page_number)
                documents.append(
                    Document(
                        page_content=summary,
                        metadata={
                            "page": page_number,
                            "modality": "visual",
                            "image_path": str(image_path),
                            "source": Path(pdf_path).name,
                        },
                    )
                )
            except Exception as error:
                print(f"Vision warning on page {page_number}:", error)

    pdf.close()
    return documents


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
EMBEDDING_DIMENSION = len(embeddings.embed_query("dimension check"))

print("Embedding model:", EMBEDDING_MODEL)
print("Embedding dimension:", EMBEDDING_DIMENSION)

pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index(PINECONE_INDEX_NAME):
    print("Creating Pinecone index...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("Waiting for Pinecone index...")
        time.sleep(2)
    print("Pinecone index created.")
else:
    print("Pinecone index already exists.")

index = pc.Index(PINECONE_INDEX_NAME)
print("Connected to:", PINECONE_INDEX_NAME)

try:
    index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
    print("Cleared namespace:", PINECONE_NAMESPACE)
except Exception as error:
    print("Namespace was probably empty. Continuing...", error)

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=PINECONE_NAMESPACE,
)


def format_context(retrieved_docs) -> str:
    parts = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        modality = doc.metadata.get("modality")
        parts.append(f"[Page {page} | {modality.upper()}]\n{doc.page_content}")
    return "\n\n".join(parts)


rag_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about the
NovaCore Systems FY2026 company report.

Use ONLY the retrieved context below.

If the answer is not available in the context, say:
"I could not find that information in the report."

Mention page numbers when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

text_rag_chain = rag_prompt | text_llm | StrOutputParser()


def answer_with_vision(question: str, context: str, image_paths: list[str | Path]) -> str:
    image_paths = image_paths[:3]
    content = [
        {
            "type": "text",
            "text": f"""
You are answering questions about the NovaCore Systems FY2026 report.

Use ONLY the retrieved context and the attached retrieved visuals.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer factually.
- Use the attached visuals when relevant.
- Mention page numbers.
- If the information is missing, say you could not find it.
""",
        }
    ]

    for image_path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_uri(image_path)},
        })

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_completion_tokens=900,
    )
    return response.choices[0].message.content.strip()


def ask_rag(question: str, show_sources: bool = True, show_images: bool = False):
    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)

    image_paths: list[str] = []
    for doc in retrieved_docs:
        if doc.metadata.get("modality") == "visual":
            image_path = doc.metadata.get("image_path")
            if image_path and Path(image_path).exists():
                image_paths.append(image_path)

    if image_paths:
        answer = answer_with_vision(
            question=question,
            context=context,
            image_paths=image_paths,
        )
    else:
        answer = text_rag_chain.invoke({"context": context, "question": question})

    print("\nQUESTION:")
    print(question)
    print("\nANSWER:")
    print(answer)

    if show_sources:
        print("\nPINECONE RETRIEVED SOURCES:")
        for i, doc in enumerate(retrieved_docs, start=1):
            print(f"{i}. Page {doc.metadata.get('page')} | {doc.metadata.get('modality')}")

    if show_images and image_paths:
        print("\nRETRIEVED VISUALS:")
        for image_path in image_paths:
            img = Image.open(image_path)
            try:
                img.show()
            except Exception:
                print(f"Open the image manually: {image_path}")

    return {
        "answer": answer,
        "documents": retrieved_docs,
        "image_paths": image_paths,
    }


def main():
    print("Extracting multimodal documents...")
    documents = extract_multimodal_documents(PDF_PATH)
    print("\nTotal LangChain documents:", len(documents))

    vectorstore.add_documents(documents)
    print("Documents uploaded to Pinecone.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("Retriever is ready.")

    demo_questions = [
        "What does NovaCore Systems do and where is the company headquartered?",
        "Which region had the highest year-over-year revenue growth?",
        "According to the revenue graph, which quarter had the highest revenue?",
        "According to the supply-chain diagram, what is the critical quality-control point?",
        "How did average support resolution time change from January to August?",
    ]

    for question in demo_questions:
        print("\n" + "=" * 80)
        ask_rag(question, show_images=True if "revenue" in question.lower() or "supply" in question.lower() else False)


if __name__ == "__main__":
    try:
        from IPython.display import display  # noqa: F401
    except Exception:  # pragma: no cover
        display = print

    # initialize retriever after vectorstore exists
    # This is a direct script port of the notebook logic.
    print("Loading project configuration...")
    print("Note: this script mirrors the notebook and may trigger model calls when run.")

    documents = extract_multimodal_documents(PDF_PATH)
    print("\nTotal LangChain documents:", len(documents))

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    EMBEDDING_DIMENSION = len(embeddings.embed_query("dimension check"))
    print("Embedding model:", EMBEDDING_MODEL)
    print("Embedding dimension:", EMBEDDING_DIMENSION)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_INDEX_NAME):
        print("Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("Waiting for Pinecone index...")
            time.sleep(2)
        print("Pinecone index created.")
    else:
        print("Pinecone index already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    print("Connected to:", PINECONE_INDEX_NAME)

    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print("Cleared namespace:", PINECONE_NAMESPACE)
    except Exception as error:
        print("Namespace was probably empty. Continuing...", error)

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )
    vectorstore.add_documents(documents)
    print("Documents uploaded to Pinecone.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("Retriever is ready.")

    sample_questions = [
        "What does NovaCore Systems do and where is the company headquartered?",
        "Which region had the highest year-over-year revenue growth?",
        "According to the revenue graph, which quarter had the highest revenue?",
    ]

    for question in sample_questions:
        print("\n" + "=" * 80)
        ask_rag(question, show_images="revenue" in question.lower())
