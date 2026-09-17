# AI Document Assistant

A simple Python RAG application that lets you ask questions about a PDF document.

The project extracts text from the PDF, splits it into chunks, stores the chunks in ChromaDB, and retrieves relevant parts when a question is asked. Gemini is then used to generate an answer based on the retrieved context.

## Technologies

* Python
* Google Gemini API
* LangChain
* ChromaDB
* pypdf

## How it works

```text
PDF
 ↓
Text extraction
 ↓
Text chunks
 ↓
Embeddings + ChromaDB
 ↓
Similarity search
 ↓
Relevant chunks
 ↓
Gemini
 ↓
Answer
```

The application also keeps the PDF page number as metadata so the retrieved information can be traced back to the source.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```powershell
pip install google-genai python-dotenv pypdf langchain-core langchain-text-splitters chromadb langchain-chroma
```

Create a `.env` file in the project folder:

```text
GEMINI_API_KEY=your_api_key_here
```

Then run:

```powershell
python pdf_assistant.py
```

## Example

```text
Ask a question: How long does the Bachelorarbeit take?

Answer:
The processing time for the Bachelorarbeit is 18 weeks.
```

## Project structure

```text
ai-document-assistant/
├── documents/
│   └── your_document.pdf
├── pdf_assistant.py
├── .env
├── .gitignore
└── README.md
```

## Note

This is a learning project built to understand the basic concepts behind Retrieval-Augmented Generation (RAG), embeddings, vector search, and LLM-based question answering.
