import os
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

class Retriever:
    def __init__(self, data_path: str = "C:/Users/nawfal/Documents/code/cooking_assistant/data",
                 model_name="BAAI/bge-small-en-v1.5",
                 k : int = 1
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chroma_path = os.path.join(data_path, 'chroma')
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs ={"normalize_embeddings": True}
        )
        db = Chroma(persist_directory=chroma_path, embedding_function=hf)
        vector_retriever = db.as_retriever(k=k)
        res = db.get()
        documents = [
            Document(page_content=txt, metadata=metadata or {})
            for txt, metadata in zip(res["documents"], res["metadatas"])
        ]
        bm25_retriever = BM25Retriever.from_documents(documents, k=k)
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.9, 0.1]
        )

    def retrieve(self, query_text: str):
        return self.hybrid_retriever.invoke(query_text)

    # def validate_retrieval(self, retrievals: list[tuple[Document, float]], threshold=0.35):
    #     for doc, score in retrievals:
    #         if score <= 0.35:
    #             return True
    #     return False

def test_retriever(k=1):
    retriever = Retriever(k=k)
    print(retriever.retrieve("give me tajine recipe"))

if __name__ == "__main__":
    test_retriever()
