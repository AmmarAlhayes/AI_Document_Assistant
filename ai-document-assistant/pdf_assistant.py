import os

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            # Clean unnecessary whitespace
            text = " ".join(text.split())

            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_number,
                    },
                )
            )

    return pages


# --------------------------------------------------
# 2. Split documents into chunks
# --------------------------------------------------

def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(pages)

    return chunks


# --------------------------------------------------
# 3. Create vector store
# --------------------------------------------------

def create_vector_store(chunks):
    vector_store = Chroma(
        collection_name="your_file_name",
    )

    vector_store.add_documents(chunks)

    return vector_store


# --------------------------------------------------
# 4. Retrieve relevant chunks
# --------------------------------------------------

def retrieve(vector_store, query, k=8):
    results = vector_store.similarity_search(
        query,
        k=k,
    )

    return results


# --------------------------------------------------
# 5. Build context for Gemini
# --------------------------------------------------

def build_context(results):
    context_parts = []

    for document in results:
        context_parts.append(
            f"Source: {document.metadata['source']}, "
            f"page {document.metadata['page']}\n"
            f"{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return context


# --------------------------------------------------
# 6. Generate answer with Gemini
# --------------------------------------------------

def generate_answer(client, query, context):
    prompt = f"""
Answer the question using only the provided context.

If the context does not contain enough information to answer,
say that you don't know.

When possible, mention the relevant source page.

Context:
{context}

Question:
{query}
"""

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
    )

    return response.output_text


# --------------------------------------------------
# 7. Main application
# --------------------------------------------------

def main():

    # Load environment variables from .env
    load_dotenv()

    # Create Gemini client
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    # PDF path
    pdf_path = "documents/your_file.pdf"

    # Load PDF
    pages = load_pdf(pdf_path)

    print("Pages loaded:", len(pages))

    # Create chunks
    chunks = split_documents(pages)

    print("Chunks created:", len(chunks))

    # Create Chroma vector store
    vector_store = create_vector_store(chunks)

    print(
        "Documents stored:",
        vector_store._collection.count()
    )

    print("\nAI Document Assistant")
    print("Type 'exit' to quit.")

    # Interactive question loop
    while True:

        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            break

        # Retrieve relevant chunks
        results = retrieve(
            vector_store,
            query,
            k=8,
        )

        # Build context
        context = build_context(results)

        # Ask Gemini
        answer = generate_answer(
            client,
            query,
            context,
        )

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()