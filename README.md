# Multimodal RAG with NovaCore FY2026 Report

A notebook-based multimodal retrieval-augmented generation (RAG) demo for the NovaCore Systems FY2026 company report. The pipeline extracts text, tables, and embedded visuals from a PDF, summarizes visuals with a Groq vision model, stores all searchable content in Pinecone, and answers questions with either a text or vision model.

## What it demonstrates

- PDF text extraction with PyMuPDF
- Table extraction and Markdown conversion
- Image and chart extraction from PDF pages
- Vision-based summaries for charts, diagrams, and images
- Hugging Face sentence embeddings
- Pinecone serverless vector search
- LangChain retrieval and text-generation chains
- Multimodal answers that include retrieved source visuals

## Project files

- `Build_Multimodal_RAG_NovaCore.ipynb`: end-to-end ingestion, indexing, retrieval, and question-answering notebook
- `NovaCore_Multimodal_Company_Report_2026.pdf`: source report used by the notebook
- `Multimodal_RAG_LangChain_Pinecone_With_Examples.pptx.pdf`: supporting presentation/reference material
- `requirements.txt`: Python dependencies

The notebook creates `novacore_extracted_images/` while processing the report.

## Requirements

- Python 3.10 or newer
- A Groq API key
- A Pinecone API key
- Jupyter Notebook, JupyterLab, or VS Code with the Jupyter extension

The embedding model is downloaded from Hugging Face on first use. Groq and Pinecone usage may incur provider charges according to your accounts.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the required API keys in the same terminal before starting Jupyter:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key"
$env:PINECONE_API_KEY = "your_pinecone_api_key"
```

Alternatively, configure these variables through your IDE or a local environment-management tool. Do not commit API keys to the repository.

## Run the notebook

1. Open `Build_Multimodal_RAG_NovaCore.ipynb`.
2. Select the virtual environment created above as the notebook kernel.
3. Confirm that `NovaCore_Multimodal_Company_Report_2026.pdf` is beside the notebook.
4. Run the cells from top to bottom.
5. Use the demo cells at the end, or call `ask_rag("your question")` with your own question.

The notebook uses these defaults:

- Text model: `openai/gpt-oss-20b`
- Vision model: `qwen/qwen3.8-27b`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Pinecone index: `novacore-multimodal-rag`
- Pinecone namespace: `fy2026-demo`

## Important behavior

The notebook deletes all vectors in the configured Pinecone namespace before uploading freshly extracted documents. Change or remove the cleanup cell if the namespace contains data you need to preserve.

The notebook uses a Groq vision request for extracted visuals. If visual processing fails for an individual image, it prints a warning and continues with the remaining text and tables.

## Pipeline

```text
PDF
  -> text, tables, and embedded images
  -> Groq vision summaries for images/charts/diagrams
  -> Hugging Face embeddings
  -> Pinecone vector index
  -> top-k retrieval
  -> text or vision answer with page/source metadata
```

## Example questions

```python
ask_rag("What does NovaCore Systems do and where is the company headquartered?")
ask_rag("Which region had the highest year-over-year revenue growth?")
ask_rag(
    "According to the revenue graph, which quarter had the highest revenue?",
    show_images=True,
)
```

## Troubleshooting

- **PDF not found:** place `NovaCore_Multimodal_Company_Report_2026.pdf` in the project root, beside the notebook.
- **Authentication errors:** verify `GROQ_API_KEY` and `PINECONE_API_KEY` are set in the notebook kernel environment.
- **Pinecone dimension errors:** use a new index or keep the index dimension aligned with the selected embedding model. The notebook detects the embedding dimension automatically when creating an index.
- **Slow first run:** model downloads, PDF visual summarization, embedding generation, and Pinecone upload all happen during ingestion.
