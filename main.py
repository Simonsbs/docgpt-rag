from app.utils.config_loader import load_config
from app.llm.ollama_client import OllamaClient
from app.vectorstore.chroma_store import VectorStore
from app.ingestion.pdf_loader import extract_text_from_pdf
from app.ingestion.chunker import chunk_text
from app.utils.history_logger import log_query

import glob
import os

def get_llm_client(config: dict):
    if config["model"] == "ollama":
        return OllamaClient()
    raise ValueError(f"Unsupported model: {config['model']}")

if __name__ == "__main__":
    config = load_config()
    llm = get_llm_client(config)
    vectorstore = VectorStore()

    pdf_dir = "./docs"
    pdf_files = glob.glob(f"{pdf_dir}/*.pdf")

    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
    else:
        for path in pdf_files:
            print(f"Processing: {path}")
            content = extract_text_from_pdf(path)
            chunks = chunk_text(content)
            base_id = os.path.splitext(os.path.basename(path))[0]

            for i, chunk in enumerate(chunks):
                doc_id = f"{base_id}_{i}"
                if not vectorstore.has_document(doc_id):
                    vectorstore.add_document(doc_id=doc_id, content=chunk)
                else:
                    print(f"Skipping already indexed: {doc_id}")


    # Query the vector DB
    query = "What is the main topic of the document?"
    print("Querying vector DB...")
    results = vectorstore.query(query)

    context = "\n".join(results)
    prompt = f"Answer the following question using this context:\n{context}\n\nQuestion: {query}"
    response = llm.ask(prompt)
    # print("LLM Response:\n", response)
    log_query(query, response)
