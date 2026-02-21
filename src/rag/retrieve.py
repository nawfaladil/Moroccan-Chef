import os
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class Retriever:
    def __init__(self, data_path: str,
                 model_name="BAAI/bge-small-en-v1.5",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chroma_path = os.path.join(data_path, 'chroma')
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs ={"normalize_embeddings": True}
        )
        self.db = Chroma(persist_directory=chroma_path, embedding_function=hf)

    def retrieve(self, query_text: str, number_of_docs=1):
        results = self.db.similarity_search_with_score(query_text, k=number_of_docs)
        if self.validate_retrieval(results):
            return results
        print("The recipe doesn't existe in our database yet.")
        return None

    def validate_retrieval(self, retrievals: list[tuple[Document, float]], threshold=0.35):
        for doc, score in retrievals:
            if score <= 0.35:
                return True
        return False
