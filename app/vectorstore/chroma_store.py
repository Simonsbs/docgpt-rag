import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="docs")
        self.embeddings = OllamaEmbeddings(model="llama3")

    def has_document(self, doc_id: str) -> bool:
        try:
            results = self.collection.get(ids=[doc_id])
            return len(results["ids"]) > 0
        except Exception:
            return False

    def add_document(self, doc_id: str, content: str):
        embedding = self.embeddings.embed_query(content)
        self.collection.add(documents=[content], embeddings=[embedding], ids=[doc_id])

    def query(self, query: str, top_k: int = 3, filter_doc: str | None = None):
        embedding = self.embeddings.embed_query(query)
        
        filter = None
        if filter_doc:
            filter = {
                "ids": {
                    "$contains": f"{filter_doc}_"
                }
            }

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filter
        )

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]

        return list(zip(ids, documents))



    def list_documents(self) -> list[str]:
        return list(set(doc_id.split("_")[0] for doc_id in self.collection.get()["ids"]))

    def remove_document(self, base_id: str):
        doc_ids = [doc_id for doc_id in self.collection.get()["ids"] if doc_id.startswith(base_id + "_")]
        self.collection.delete(ids=doc_ids)
